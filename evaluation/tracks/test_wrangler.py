import multiprocessing
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ExaTrkXDataIO import DataReader

import matplotlib.pyplot as plt

from plot_configurations import (
    particle_filters,
)
from functools import partial
import os
from typing import List, Any, Tuple


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch
from track_reconstruction.algorithm.track_reco_algorithm import TrackRecoAlgorithm 
import networkx as nx
import operator
 
class WranglerTrackReco(TrackRecoAlgorithm):
    def __init__(self, filter_cut:float = 0.01, walk_min: float = 0.3, walk_max: float = 0.8, cc_cut: float = 0.01):
        self.filter_cut = filter_cut
        self.walk_min = walk_min 
        self.walk_max = walk_max 
        self.cc_cut = cc_cut
        self.score_name = "edges_scores"
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
        
    
    def build_roads(self, starting_node, used_hits: set) -> List[Tuple]:
        
        path = [(starting_node,)]        
        while True:
            new_path = []
            is_all_none = True

            for pp in path:
                start = pp[-1]
                if start is None:
                    new_path.append(pp)
                    continue     
                current_used_hits = used_hits | set(pp)
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
        
        used_nodes = set()
        sub_graphs = []
        
        for node in nx.topological_sort(self.G):
            if node in used_nodes:
                continue
            
            road = self.build_roads(node, used_nodes)
            a_road = max(road, key = len)[:-1]
            
            if len(a_road) < 3:
                used_nodes.add(node)
                continue
            sub_graphs.append([self.G.nodes[n]["hit_id"] for n in a_road])
            used_nodes.update(a_road)
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
        edge_index[edge_flip_mask, :] = edges[edge_flip_mask, ::-1] 
        edge_index = torch.from_numpy(edge_index).long()
        R = R.unsqueeze(1)
        hit_id = hit_id.unsqueeze(1)
        
        graph = Data(x = R, edge_index = edge_index.T, edges_scores = score, hit_id = hit_id, num_nodes = n_hits)
        self.G = to_networkx(graph, ["hit_id"], [self.score_name], to_undirected=False) 
        
        list_fake_edges = [
            (u, v) for u, v, e in self.G.edges(data = True) if e[self.score_name] <= self.cc_cut 
        ] 
        self.G.remove_edges_from(list_fake_edges)
        self.G.remove_nodes_from(list(nx.isolates(self.G))) 
        
        trkx = self.get_tracks()
        
        for trk in trkx:
            trk.sort(key = lambda x: R[x])
        data = []
        
        for group_id, hit_list in enumerate(trkx):
            for hit_id in hit_list:
                data.append({'hit_id': hit_id, 'track_id': group_id})
        
        results = pd.DataFrame(data)
        results['track_id'] =  results.track_id.astype(int)
        track_candidates = np.array([
            item for track in trkx for item in [*track, -1]
        ])
        return results

def analyze_tracks(truth, submission):
    """
    Taken from https://github.com/LAL/trackml-library/blob/master/trackml/score.py
    Compute the majority particle, hit counts, and weight for each track.
    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.
    Returns
    -------
    pandas.DataFrame
        Contains track_id, nhits, major_particle_id, major_particle_nhits,
        major_nhits, and major_weight columns.
    """
    particles_nhits = truth['particle_id'].value_counts(sort = False)
    event = pd.merge(
        truth[['hit_id', 'particle_id']],
        submission[['hit_id', 'track_id']],
        on=['hit_id'],
        how='left',
        validate='one_to_one'
    )
    print(event[event['particle_id'] == 103582794466197504]) 
    
    event.drop(
        'hit_id',
        axis=1,
        inplace=True
    )
    event.sort_values(
        by=['track_id', 'particle_id'],
        inplace=True
    )
    
    tracks = []
    # running sum for the reconstructed track we are currently in
    rec_track_id = -1
    rec_nhits = 0
    # running sum for the particle we are currently in (in this track_id)
    cur_particle_id = -1
    cur_nhits = 0
    # majority particle with most hits up to now (in this track_id)
    maj_particle_id = -1
    maj_nhits = 0
    for hit in event.itertuples(index=False):
        # we reached the next track so we need to finish the current one
        if (rec_track_id != -1) and (rec_track_id != hit.track_id):
            # could be that the current particle is the majority one
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
            # store values for this track
            tracks.append((rec_track_id, rec_nhits, maj_particle_id,
                           particles_nhits[maj_particle_id], maj_nhits))

        # setup running values for next track (or first)
        if rec_track_id != hit.track_id:
            rec_track_id = hit.track_id
            rec_nhits = 1
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            maj_particle_id = -1
            maj_nhits = 0
            continue

        # hit is part of the current reconstructed track
        rec_nhits += 1

        # reached new particle within the same reconstructed track
        if cur_particle_id != hit.particle_id:
            # check if last particle has more hits than the majority one
            # if yes, set the last particle as the new majority particle
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
            # reset runnig values for current particle
            cur_particle_id = hit.particle_id
            cur_nhits = 1
        # hit belongs to the same particle within the same reconstructed track
        else:
            cur_nhits += 1

    # last track is not handled inside the loop
    if maj_nhits < cur_nhits:
        maj_particle_id = cur_particle_id
        maj_nhits = cur_nhits

    if rec_track_id != -1:
        # store values for the last track
        tracks.append((
            rec_track_id,
            rec_nhits,
            maj_particle_id,
            particles_nhits[maj_particle_id],
            maj_nhits
        ))

    cols = [
        'track_id',
        'nhits',
        'major_particle_id',
        'major_particle_nhits',
        'major_nhits'
    ]

    return pd.DataFrame.from_records(tracks, columns=cols)
    
     
def match_tracks(
    truth: pd.DataFrame,
    reconstructed: pd.DataFrame,
    particles: pd.DataFrame,
    min_pt=1.,
    frac_reco_matched=0.5,
    frac_truth_matched=0.5,
    particle_filter = None
):
    """
    Match reconstructed tracks to particles.

    Args:
        truth: a dataframe with columns of ['hit_id', 'particle_id']
        reconstructed: a dataframe with columns of ['hit_id', 'track_id']
        particles: a dataframe with columns of
            ['particle_id', 'pt', 'eta', 'radius', 'vz', 'charge'].
            radius = sqrt(vx**2 + vy**2),
            ['vx', 'vy', 'vz'] are the production vertex of the particle
        min_hits_truth: minimum number of hits for truth tracks
        min_hits_reco:  minimum number of hits for reconstructed tracks

    Returns:
        A tuple of (
            n_true_tracks: int, number of true tracks
            n_reco_tracks: int, number of reconstructed tracks
            n_matched_reco_tracks: int, number of reconstructed tracks
                matched to true tracks
            matched_pids: np.narray, a list of particle IDs matched
                by reconstructed tracks
        )
    """
    # just in case particle_id == 0 included in truth.
    truth = truth[truth.particle_id > 0]

    # Associate hits with particle data.
    hits = truth.merge(
        particles,
        on='particle_id',
        how='left'
    )

    # Count number of hits for each particle.
    n_hits_per_particle = hits.groupby("particle_id")['hit_id'].count()
    n_hits_per_particle = n_hits_per_particle.reset_index().rename(columns={
        "hit_id": "nhits"
    })
    # Associate number of hits to hits.
    hits = hits.merge(n_hits_per_particle, on='particle_id', how='left')
    if min_pt > 0:
        hits = hits[
            (hits.pt >= min_pt)
        ]
    hits = hits[hits.eta.abs() <= 3]

    # Extract trackable particles.
    trackable_pids = np.unique(
        hits.particle_id.values
    )
    #print(len(trackable_pids))
    pruned_sub = reconstructed[
        reconstructed.hit_id.isin(hits.hit_id)
    ]

    # some hits do not exist in the reconstructed tracks,
    # fill their track id with -1.
    if pruned_sub.shape[0] != hits.shape[0]:
        extended_sub = hits[['hit_id']].merge(
            pruned_sub,
            on='hit_id',
            how='left'
        ).fillna(-1)
    else:
        extended_sub = pruned_sub
    #print(extended_sub.track_id)
    extended_sub.track_id = extended_sub.track_id.astype(int)
    # Compute track properties.
    tracks = analyze_tracks(hits, extended_sub)
    
    # double-majority matching criteria.
    purity_rec = np.true_divide(
        tracks['major_nhits'],
        tracks['nhits']
    )
    purity_maj = np.true_divide(
        tracks['major_nhits'],
        tracks['major_particle_nhits']
    )
    matched_reco_track = (
        (frac_reco_matched < purity_rec) &
        (frac_truth_matched < purity_maj)
    )

    reco_tracks = tracks
    matched_pids = tracks[matched_reco_track].major_particle_id.values

    n_true_tracks = np.unique(hits.particle_id).shape[0]
    n_reco_tracks = reco_tracks.shape[0]
    n_matched_reco_tracks = np.sum(matched_reco_track)

    is_matched = particles.particle_id.isin(matched_pids).values
    is_trackable = particles.particle_id.isin(trackable_pids).values

    particles = particles.assign(
        is_matched=is_matched,
        is_trackable=is_trackable
    )

    # Only track particle with charge.
    # This should be common requirement.
    composed_filter = (particles.charge.abs() > 0)

    # Custom filter.
    if particle_filter is not None:
        composed_filter &= particle_filter

    # pT cut.
    if min_pt > 0:
        composed_filter &= (particles.pt >= min_pt)

    particles = particles[
        composed_filter
    ]

    return (
        n_true_tracks,
        n_reco_tracks,
        n_matched_reco_tracks,
        particles
    )


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
    #particles = particles[particle_filters['HSS'](particles)]
    #print(data['hits']['particle_id'])
    id_map = {old_id: new_id for new_id, old_id in enumerate(data['hits']['hit_id'].unique())} 
    data['hits']['hit_id'] = data['hits']['hit_id'].map(id_map)
    data['edges']['sender'] = data['edges']['sender'].map(id_map)
    data['edges']['receiver'] = data['edges']['receiver'].map(id_map)
    score = data['edges']['score'].to_numpy() 
    edges = data['edges'][['sender', 'receiver']].to_numpy() 
    
    constructed_tracks = reconstruct_tracks(
        algorithm = WranglerTrackReco(),
        hits = data['hits'][['hit_id', 'r']].to_numpy(),
        edges = edges,
        score = score
    ) 
    #raise ValueError(data['hits'], constructed_tracks) 
    n_true_tracks, n_reco_tracks, n_matched_reco_tracks, particles = match_tracks(
        truth=data['hits'],
        reconstructed=constructed_tracks,
        particles=particles,
        min_pt=1.
    )
    if statistics is True:
        return particles, {
            'n_true': n_true_tracks,
            'n_reco': n_reco_tracks,
            'n_match': n_matched_reco_tracks
        }
    else:
        print('n_true: ', n_true_tracks,
            'n_reco: ', n_reco_tracks,
            'n_match: ',n_matched_reco_tracks)
        
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
    path = Path('HSSPU0-ModelHNLPU200-GNN.yaml')
    extract()
    
    
    
if __name__ == '__main__':
    main()