# -*- coding: utf-8 -*-
"""Data Reduction Algorithms.

This module provides implementations for data reduction algorithms.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class EGDIS(object):
    """The enhanced global density-based instance selection algorithm. Cited: Malhat M, El Menshawy M, Mousa H,
    et al. A new approach for instance selection: Algorithms, evaluation, and comparisons[J]. Expert Systems with
    Applications, 2020, 149: 113297. """

    def __init__(self, k=4):
        self.k = k
        self.x = None
        self.y = None
        self.neigh_dist = None
        self.neigh_ind = None
        self.neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)

    def _irrelevance(self, x_kneighbors):
        score = 0
        for inx in range(1, self.k):
            if self.y[x_kneighbors[0]] != self.y[x_kneighbors[inx]]:
                score += 1
        return score

    def _densest(self, unique_x, size):
        scores, _ = self.neigh.kneighbors(self.x[unique_x], size)
        return -scores.sum(axis=1) / len(scores)

    def _check_densest(self, x_kneighbors, dense_scores, unique_x):
        maximum = dense_scores[np.where(unique_x == x_kneighbors[0])[0]]
        for idx in range(1, self.k):
            if dense_scores[np.where(unique_x == x_kneighbors[idx])[0]] > maximum:
                return False
        return True

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.neigh.fit(x)
        self.neigh_dist, self.neigh_ind = self.neigh.kneighbors(x)

        # start the EGDIS algorithm
        # calculate the irrelevance of each sample, 0 means no neighbors have different labels.
        irrelevance_scores = np.array([self._irrelevance(x_kneighbors) for x_kneighbors in self.neigh_ind])
        dense_x = self.neigh_ind[irrelevance_scores == 0]
        unique_x = np.unique(dense_x)

        # calculate the dense scores for unique and irrelevance_scores == 0 samples.
        size = len(self.x)
        dense_scores = self._densest(unique_x, size)
        selected = [self._check_densest(x_kneighbors, dense_scores, unique_x) for x_kneighbors in dense_x]
        selected = dense_x[selected]
        selected = [x[0] for x in selected]

        # select from the boundary samples.
        selected_boundary = self.neigh_ind[irrelevance_scores >= int(self.k / 2)]
        selected_boundary = [x[0] for x in selected_boundary]

        return selected, selected_boundary

    def reduce(self, percent):

        return
