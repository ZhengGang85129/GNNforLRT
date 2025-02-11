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
from typing import List, Any, Tuple

from tracks.track_matching import match_tracks, analyze_tracks

import numpy as np
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch
from tracks.track_reconstruction.algorithm.track_reco_algorithm import TrackRecoAlgorithm 
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
                       used_hits: set) -> List[Tuple]:
        
    
        
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
        
    
    def build_roads(self, starting_node) -> List[Tuple]:
        
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
    def get_tracks(self):
        
        self.used_nodes = set()
        sub_graphs = []

        for node in nx.topological_sort(self.G):
            if node in self.used_nodes:
                continue
            
            road = self.build_roads(node)
            a_road = max(road, key = len)[:-1]

            if len(a_road) < 3:
                self.used_nodes.add(node)
                continue
            
            sub_graphs.append([self.G.nodes[n]["hit_id"] for n in a_road])
            self.used_nodes.update(a_road)
        return sub_graphs

    def reconstruct(
        self,
        hits: np.array,
        edges: np.array,
        score: np.array
    ) -> pd.DataFrame:
        
        #print(hits) 
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
        #raise ValueError(R[])
        #is_acyclic = nx.is_directed_acyclic_graph(self.G) 
        #if not is_acyclic: 
            #return 
            #cycles = list(nx.simple_cycles(self.G))
            #num_cycles = len(cycles)
            
            #raise ValueError(num_cycles, cycles, R[11790], R[11791])
        list_fake_edges = [(u, v) for u, v, e in self.G.edges(data = True) if e[self.score_name] <= self.cc_cut ] 
        graph_lock = Lock()
        with graph_lock:
            self.G.remove_edges_from(list_fake_edges)
            self.G.remove_nodes_from(list(nx.isolates(self.G))) 
        cycles = list(nx.simple_cycles(self.G))
        if cycles:
            print(f"Found {len(cycles)} cycles after modifications")
            print(f"=> Skip this event...") 
            return pd.DataFrame(data=[], columns=['hit_id', 'track_id']) 
        trkx = self.get_tracks()
        for trk in trkx:
            trk.sort(key = lambda x: R[x])
        data = []
        
        for group_id, hit_list in enumerate(trkx):
            for hit_id in hit_list:
                data.append({'hit_id': hit_id, 'track_id': group_id})
        results = pd.DataFrame(data)
        track_candidates = np.array([
            item for track in trkx for item in [*track, -1]
        ])
        results['track_id'] =  results.track_id.astype(int)
        return results

    
     


def reconstruct_tracks(
    algorithm: TrackRecoAlgorithm,
    hits: np.array,
    edges: np.array,
    score: np.array,
):
    """
    Reconstruct tracks.

    :param algorithm: Reconstruction algorithm.
    :param hits: 1xN array, hit ID.
    :param edges: 2xN array, first hit index, second hit index in hits.
    :param score: 1xN array, score for each edge.
    :param edge_filter: Filter apply to edges.

    :return:
    """

    tracks = algorithm.reconstruct(
        hits, edges, score
    )

    return tracks


def reconstruct_and_match_tracks(
    data,
    statistics = False 
):
    
    '''
    data: particles/edges/hits/
    edges: sender/receiver/truth/score
    particles: particle_id/particle_type/charge/parent_ptype/vx/vy/vz/px/py/pz
    hits:hit_id/particle_id/r/phi/z
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
    
    #id_map = {old_id: new_id for new_id, old_id in enumerate(data['hits']['hit_id'].unique())} 
    #data['hits']['hit_id'] = data['hits']['hit_id'].map(id_map)
    #data['edges']['sender'] = data['edges']['sender'].map(id_map)
    #data['edges']['receiver'] = data['edges']['receiver'].map(id_map)
    score = data['edges']['score'].to_numpy() 
    edges = data['edges'][['sender', 'receiver']].to_numpy() 
    
    constructed_tracks = reconstruct_tracks(
        algorithm = WranglerTrackReco(),
        hits = data['hits'][['hit_id', 'R']].to_numpy(),
        edges = edges,
        score = score
    ) 
    n_true_tracks, n_reco_tracks, n_matched_reco_tracks, particles = match_tracks(
        truth=data['hits'],
        reconstructed=constructed_tracks,
        particles=particles,
        min_pt=1.
    )
    
    if statistics is True:
        return particles, {
            'n_true: ': n_true_tracks,
            ', n_reco: ': n_reco_tracks,
            ', n_match: ': n_matched_reco_tracks,
            ', effeciency: ': n_matched_reco_tracks/n_true_tracks
        }
    else:
        #print('n_true: ', n_true_tracks,
        #    ', n_reco: ', n_reco_tracks,
        #    ', n_match: ',n_matched_reco_tracks,
        #    ', efficiency: ', n_matched_reco_tracks/n_true_tracks 
        #    )
        
        return particles

def extract():
    with multiprocessing.Pool(processes = 8) as pool:
        reader = DataReader(
            config_path = path,
            base_dir = './'
        )
        particles = pool.map(partial(reconstruct_and_match_tracks), reader.read())
    
    
def main():
    global path
    path = Path('./tracks/DBSCAN_config/TTBar_PU200.yaml')
    extract()
    
    
    
if __name__ == '__main__':
    main()