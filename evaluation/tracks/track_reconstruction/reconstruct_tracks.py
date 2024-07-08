#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .algorithm import TrackRecoAlgorithm


def reconstruct_tracks(
    algorithm: TrackRecoAlgorithm,
    hits: np.array,
    edges: np.array,
    score: np.array,
    edge_filter=None
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
    if edge_filter is not None:
        edges = edges[edge_filter]
        score = score[edge_filter]

    tracks = algorithm.reconstruct(
        hits, edges, score
    )

    return tracks
