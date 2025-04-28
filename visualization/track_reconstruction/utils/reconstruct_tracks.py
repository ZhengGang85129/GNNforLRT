#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from track_reconstruction.algorithm.track_reco_algorithm import TrackRecoAlgorithm


def reconstruct_tracks(
    algorithm: TrackRecoAlgorithm,
    hits: np.array,
    edges: np.array,
    score: np.array,
    edge_filter = None
):
    """
    Reconstruct tracks.
    
    Args:
        algorithm: Reconstruction algorithm.
        hits: Array of hit id. Expected shape (n_hits, 1).
        edges(np.array): Array of candidate edges between hits. Each row contains (source, target).
        score(np.array): Array of edge scores corresponding to each edge.
    Return:
        pd.DataFrame: A DataFrame containing reconstructed tracks with columns:
        - 'hit_id': ID of hit
        - 'track_id': Assigned track id.
    
    
    """
    if edge_filter is not None:
        edges = edges[edge_filter]
        score = score[edge_filter]

    tracks = algorithm.reconstruct(
        hits, edges, score
    )

    return tracks
