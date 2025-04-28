#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy as sp

from sklearn.cluster import DBSCAN

import multiprocessing

import numpy as np
import pandas as pd
from ExaTrkXDataIO import DataReader

from track_reconstruction.utils.reconstruct_tracks import reconstruct_tracks
from track_reconstruction.utils.match_tracks import match_tracks
from .track_reco_algorithm import TrackRecoAlgorithm


class DBSCANTrackReco(TrackRecoAlgorithm):
    '''
    Track Reconstruction algorithm based on DBSCAN clustering.
    
    Args:
        epsilon(float, optional): The maximum distance between two samples for them to be considered as in the same neighborhood. Defaults: 0.25.
        min_samples (int, optional): The number of samples in a neighborhood for a point to be considered as core point. Defaults to 2.
    '''
    
    def __init__(self, epsilon: float =0.25, min_samples: int = 2):
        self.epsilon = epsilon
        self.min_samples = min_samples

    def reconstruct(
        self,
        hits: np.array,
        edges: np.array,
        score: np.array
    ) -> pd.DataFrame:
        """
        Reconstruct tracks.
        
        Args:
            hits(np.array): Array of hit id. Expected shape (n_hits,1).
            edges(np.array): Array of candidate edges between hits. Each row contains (source, target).
            score(np.array): Array of edge scores corresponding to each edge.
        Return:
            pd.DataFrame: A DataFrame containing reconstructedd tracks with columns:
            - 'hit_id': ID of hit
            - 'track_id': Assigned track id.
        """
        
        
        n_hits = hits.shape[0]

        # Prepare the DBSCAN input, which the adjacency matrix
        # with its value being the edge score.
        score_matrix = sp.sparse.csr_matrix(
            (score, (edges[:, 0], edges[:, 1])),
            shape=(n_hits, n_hits),
            dtype=np.float32
        )

        # Rescale the duplicated edges
        score_matrix.data[
            score_matrix.data > 1
        ] /= 2.0

        # Invert to treat score as an inverse distance
        score_matrix.data = 1 - score_matrix.data

        # Make it symmetric
        symmetric_score_matrix = sp.sparse.coo_matrix((
            np.hstack([
                score_matrix.tocoo().data,
                score_matrix.tocoo().data
            ]),
            np.hstack([
                np.vstack([
                    score_matrix.tocoo().row,
                    score_matrix.tocoo().col
                ]),
                np.vstack([
                    score_matrix.tocoo().col,
                    score_matrix.tocoo().row
                ])
            ])
        ))

        # Apply DBSCAN.
        clustering = DBSCAN(
            eps=self.epsilon,
            metric='precomputed',
            min_samples=self.min_samples
        ).fit_predict(symmetric_score_matrix)

        # Only consider hits which have score larger then zero.
        hit_index = np.unique(symmetric_score_matrix.tocoo().row)
        track_id = clustering[hit_index]
        return pd.DataFrame(data={
            "hit_id": hits[hit_index],
            "track_id": track_id
        })
        
        
def reconstruct_and_match_tracks(
    data,
    epsilon=0.20,
    statistics=False
):
    """

    :param data: Data to reconstruct track candidates
    :param epsilon: DBSCAN epsilon
    :param statistics: Whether return statistics of result.
    :return:
    """

    particles = data['particles']
    particles = particles.drop_duplicates(subset=[
        'particle_id'
    ])

    # Compute some necessary parameters.
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
    # Reconstruct tracks.
    constructed_tracks = reconstruct_tracks(
        algorithm=DBSCANTrackReco(
            epsilon=epsilon,
            min_samples=2
        ),
        hits=data['hits']['hit_id'].to_numpy(),
        edges=data['edges'][['sender', 'receiver']].to_numpy(),
        score=data['edges']['score'].to_numpy()
    )

    mean_track_length = constructed_tracks['track_id'].value_counts().mean()

    # Match track to truth label.
    n_true_tracks, n_reco_tracks, n_matched_reco_tracks, particles, ghost_rate, mean_track_length, matched_track = match_tracks(
        truth=data['hits'],
        reconstructed=constructed_tracks,
        particles=particles,
        min_hits_truth=5,
        min_hits_reco=5,
        min_pt=1.
        # ITK dataset requirement.
        # particle_filter=particle_filter
    )

    if statistics is True:
        return particles, {
            'hits': data['hits'],
            'constructed_tracks': constructed_tracks,
            'edges': data['edges'],
            'matched_track': matched_track
        }
    else:
        return particles, {
            'ghost_rate': ghost_rate,
            'mean_track_length': mean_track_length
        }


def reconstruct_and_match_tracks_with_reader(reader: DataReader):
    """
    Multi-process reading.

    :param reader: Reader configured to read GNN output result.
    :return:
    """
    with multiprocessing.Pool(processes=8) as pool:
        particles = pd.concat(
            pool.map(reconstruct_and_match_tracks, reader.read())
        )

    return particles
