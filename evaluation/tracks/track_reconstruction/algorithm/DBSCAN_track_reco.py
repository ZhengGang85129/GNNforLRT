#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy as sp

from sklearn.cluster import DBSCAN

from .track_reco_algorithm import TrackRecoAlgorithm


class DBSCANTrackReco(TrackRecoAlgorithm):
    def __init__(self, epsilon=0.25, min_samples=2):
        self.epsilon = epsilon
        self.min_samples = min_samples

    def reconstruct(
        self,
        hits: np.array,
        edges: np.array,
        score: np.array
    ) -> pd.DataFrame:
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
