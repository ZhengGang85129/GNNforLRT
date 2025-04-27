#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class TrackRecoAlgorithm(ABC):
    @abstractmethod
    def reconstruct(
        self,
        hits: np.array,
        edges: np.array,
        score: np.array
    ) -> pd.DataFrame:
        """
        Reconstruct tracks.

        :param hits: 1xN array, hit ID.
        :param edges: 2xN array, first hit index, second hit index in hits.
        :param score: 1xN array, score for each edge.

        :return: Dataframe with hit_id and track_id columns.
        """
        pass
