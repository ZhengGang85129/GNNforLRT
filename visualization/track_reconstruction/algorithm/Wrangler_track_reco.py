#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import multiprocessing
from multiprocessing import Lock
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ExaTrkXDataIO import DataReader

import matplotlib.pyplot as plt

from functools import partial
import os
from typing import List, Any, Tuple, Optional

from track_reconstruction.utils.match_tracks import match_tracks

import numpy as np
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch
from .track_reco_algorithm import TrackRecoAlgorithm 
from track_reconstruction.utils.reconstruct_tracks import reconstruct_tracks
import networkx as nx
import operator
 
class WranglerTrackReco(TrackRecoAlgorithm):
    def __init__(self, filter_cut:float = 0.1, walk_min: float = 0.1, walk_max: float = 0.6, cc_cut: float = 0.01):
        self.filter_cut = filter_cut
        self.walk_min = walk_min 
        self.walk_max = walk_max 
        self.cc_cut = cc_cut
        self.score_name = "edges_scores"
        assert self.walk_min <= self.walk_max
    
    def find_next_hits(self,
                       current_hit: int,
                       used_hits: set) -> Optional[List[Tuple]]:
        """
        Finds the next candidates to extend a track from the current hit.
        
        Args:
            current_hit (int): Current hit index from which to search
            used_hits (Set[int]): Set of hits that have been used
        Return:
            Optional[List[int]]: List of next hit id to extend the track, or None if no valid continuation is found.
        """

        
        neighbors = [ n for n in self.G.neighbors(current_hit) if n not in used_hits]
        if not neighbors: return None

        neighbors_scores = [(n, self.G.edges[(current_hit, n)][self.score_name]) for n in neighbors]
        
        best_neighbor = max(neighbors_scores, key=operator.itemgetter(1))
      
        if best_neighbor[1] <= self.walk_min:
            return None

        # Always add the neighoors with a score above th_add
        next_hits = [n for n, score in neighbors_scores if score > self.walk_max]

        # Always add the highest scoring neighbor above th_min if it's not already in next_hits
        if not next_hits:
            best_hit = best_neighbor[0]
            if best_hit not in next_hits:
                next_hits.append(best_hit)

        return next_hits
        
    
    def build_roads(self, starting_node:int) -> List[Tuple]:
        '''
        Constructs all possible paths starting from a given node
        Args:
            starting_node (int): Starting hit index.
        Returns:
            List[Tuple[int, ...]]: List of candidate paths (tuples of hit ids).
        '''
        path = [(starting_node,)]        
        while True:
            new_path = []
            is_all_none = True

            for pp in path:
                start = pp[-1]
                if start is None:
                    new_path.append(pp)
                    continue     
                current_used_hits = self.used_nodes | set(pp)
                next_hits = self.find_next_hits(start, current_used_hits)
                if next_hits is None:
                    new_path.append((*pp, None))
                else:
                    is_all_none = False
                    new_path.append((*pp, *next_hits))
            path = new_path
            if is_all_none:
                break
        return path
    
    
    def get_tracks(self) -> List[List[int]]:
        '''
        Construct particle tracks from the directed acyclic graph(DAG).
        
        Traverses the graph in topological order to build tracks by following the longest valid path starting from each unused node.
        
        Return:
            List[List[int]]: A List of reconstructed tracks, where each track is a list of hit IDs (integers).
        '''
        self.used_nodes = set()
        sub_graphs = []

        for node in nx.topological_sort(self.G):
            if node in self.used_nodes:
                continue
            
            road = self.build_roads(node)
            a_road = max(road, key = len)[:-1] # Find the longest path

            if len(a_road) < 3: # minimum n_hits threshold for a track
                self.used_nodes.add(node)
                continue
            
            #converting graph node indices to hit ids
            sub_graphs.append([self.G.nodes[n]["hit_id"] for n in a_road])
            self.used_nodes.update(a_road)
            
            
        return sub_graphs

    def reconstruct(
        self,
        hits: np.array,
        edges: np.array,
        score: np.array
    ) -> pd.DataFrame:
        '''
        Reconstruct tracks from a set of hits, edges, and associated scores.
        Args:
            hits(np.array): Array of hit information. Expected shape (n_hits, n_features). Columns: (hit_id, R)
            edges(np.array): Array of candidate edges between hits. Each row contains (source, target).
            score(np.array): Array of edge scores corresponding to each edge.
        Return:
            pd.DataFrame: A DataFrame containing reconstructed tracks with columns:
            - 'hit_id': ID of hit
            - 'track_id': Assigned track id (seed-based)
            - 'seed_id': The seed hit ID associated with the track
        '''
        
        
        n_hits = hits.shape[0]
        hit_id = torch.from_numpy(hits[:, 0]).long()
        edges = edges[np.where(score > self.filter_cut)[0]]
        score = torch.from_numpy(score[np.where(score > self.filter_cut)[0]])
        R = torch.from_numpy(hits[:, 1])
        
        edge_flip_mask = R[edges[:, 0]] > R[edges[:, 1]]
        edge_index = edges.copy()
        edge_index[edge_flip_mask, :] = edge_index[edge_flip_mask, ::-1]
        edge_index = torch.from_numpy(edge_index).long()
        
        graph = Data(x = R, edge_index = edge_index.T, edges_scores = score, hit_id = hit_id, num_nodes = n_hits)
        self.G = to_networkx(graph, ["hit_id"], [self.score_name], to_undirected = False) 
        list_fake_edges = [(u, v) for u, v, e in self.G.edges(data = True) if e[self.score_name] <= self.cc_cut ] 
        graph_lock = Lock()
        with graph_lock:
            self.G.remove_edges_from(list_fake_edges)
            self.G.remove_nodes_from(list(nx.isolates(self.G))) 
        while True:
            try:
                #print("Graph type:", type(self.G))
                cycle = nx.find_cycle(self.G, orientation="original")
                print(cycle[0])
                for edge in cycle:
                    u, v = edge[0], edge[1]
                    self.G.remove_edge(u, v)
                    print(f"Removed edge {u} -> {v}")
            except nx.NetworkXNoCycle:
                break
                    
        trkx = self.get_tracks()
        seeds = []
        for trk in trkx:
            trk.sort(key = lambda x: R[x])
        seeds = [seed[0] for seed in trkx]    
        data = []
        assert len(seeds) == len(trkx) 
        for group_id, (hit_list, seed) in enumerate(zip(trkx, seeds)):
            for hit_id in hit_list:
                data.append({'hit_id': hit_id, 'track_id': str(seed)+'-trk', 'seed_id': seed})
        results = pd.DataFrame(data)
        track_candidates = np.array([
            item for track in trkx for item in [*track, -1]
        ])
        results['track_id'] =  results.track_id.astype(str)
        results['seed_id'] =  results.seed_id.astype(int)
        return results

def reconstruct_and_match_tracks(
    data:pd.DataFrame,
    filter_cut: float = 0.1,
    walk_min: float = 0.3,
    walk_max: float = 0.8,
    return_statistics = False 
):
    
    '''
    Reconstructs particle tracks from a set of hits and edges, and optionally evaluates matching statistics.
    
    Args:
        data(pd.DataFrame): A DataFrame containing information about hits, edges, and associated particle IDs.
        filter_cut(float, optional): Minimum edge score threshold to retain an edge. Defaults to 0.1.
        walk_min(float, optional): Minimum edge score to consider a candidate for track walking. Defaults to 0.3.
        walk_max(float, optional): Minimum edge score for selecting a strongest candidate during walking. Defaults to 0.8.
        return_statistics(bool, optional): Whether to calculate and output matching statistics. Defaults to False.
    
    Returns:
        tuple:
            A tuple (particle, result_dict), where:
            - particles (pd.DataFrame): A DataFrame of matched particles.
            - result_dict (dict):
                - If `return_statistics` is True: contains
                    - 'hits': DataFrame of hit features
                    - 'constructed_tracks': DataFrame of constructed tracks
                    - 'edges': DataFrame of sender, receiver, score, and truth of constructed edges.
                    - 'matched_tracks': Matching information between true and reconstructed tracks.
                    - 'ghost_rate': Ratio of ghost hits in the matched tracks.
                    - 'mean_track_length': Average length of reconstructed tracks.
    
    '''
    #particles = 
    particles = data['particles']
    particles = particles.drop_duplicates(subset = ['particle_id'])
    pt = np.sqrt(particles['px'] ** 2 + particles['py'] ** 2)
    pz = particles['pz']
    z0 = particles['vz']
    d0 = np.sqrt(particles['vx']**2 + particles['vy']**2)
    p3 = np.sqrt(pt ** 2 + pz ** 2)
    p_theta = np.arccos(pz / p3)
    eta = -np.log(np.tan(0.5 * p_theta))
    vr = np.sqrt(d0**2 + particles['vz']**2)
    particles = particles.assign(
        pt=pt,
        eta=eta,
        z0=z0,
        d0=d0,
        vr=vr
    )
    particles = particles[particles.eta.abs() <=3]
    hit_r = np.sqrt((data['hits']['r']) ** 2 + (data['hits']['z']) ** 2)
    
    data['hits'] = data['hits'].assign(R = hit_r) 
    
    score = data['edges']['score'].to_numpy() 
    edges = data['edges'][['sender', 'receiver']].to_numpy() 
    
    constructed_tracks = reconstruct_tracks(
        algorithm = WranglerTrackReco(filter_cut = filter_cut, walk_min = walk_min, walk_max = walk_max),
        hits = data['hits'][['hit_id', 'R']].to_numpy(),
        edges = edges,
        score = score
    ) 
    n_true_tracks, n_reco_tracks, n_matched_reco_tracks, particles, ghost_rate, mean_track_length, matched_track = match_tracks(
        truth=data['hits'],
        reconstructed=constructed_tracks,
        particles=particles,
        min_pt=1.
    )
    
    if return_statistics is True:
        return particles, {
            'hits': data['hits'],
            'constructed_tracks': constructed_tracks,
            'edges': data['edges'],
            'matched_track': matched_track,
            'ghost_rate': ghost_rate,
            'mean_track_length': mean_track_length
        }
    else:
        return particles 