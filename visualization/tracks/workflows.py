#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A collection of some track construction workflows,
which are composed with some of elementary operations.
"""

import multiprocessing

import numpy as np
import pandas as pd
from ExaTrkXDataIO import DataReader

from track_reconstruction import reconstruct_tracks
from track_reconstruction.algorithm import DBSCANTrackReco
from track_matching import match_tracks


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

    # We can save intermediate result here.
    """
    np.savez(
        f'../../data/models/track_candidates/{data.gnn_arch}/test/{data.evtid}.npz', {
            'hit_id': constructed_tracks['hit_id'],
            'track_id': constructed_tracks['track_id']
        }
    )
    """
    #print(constructed_tracks)
    #ghost_rate = len(constructed_tracks['hit_id'][constructed_tracks['track_id'] == -1])/len(constructed_tracks['track_id'])
    mean_track_length = constructed_tracks['track_id'].value_counts().mean()
    #print(ghost_rate, mean_track_length)
    # ITK dataset requirement.
    """
    particle_filter = (
        (particles.status == 1) &
        (particles.barcode < 200000) &
        (particles.radius < 260)
    )
    """

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
