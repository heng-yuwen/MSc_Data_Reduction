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
        """Init a new EGDIS instance
        :param k: the nearest neighbor parameter, should be k+1 if the instance is in the dataset.
        """

        self.k = k
        self.x = None
        self.y = None
        self.neigh_dist = None
        self.neigh_ind = None
        self.neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1)

    def _irrelevance(self, x_kneighbors):
        """Count the number of samples with different labels around x
        :param x_kneighbors: the array of neighbor indices.
        """

        score = 0
        for inx in range(1, self.k):
            if self.y[x_kneighbors[0]] != self.y[x_kneighbors[inx]]:
                score += 1
        return score

    def _dense(self, unique_x, size):
        """Calculate the dense score for sample x. The equation is in the paper.
        :param unique_x: the array of unique x indices.
        :param size: the size of the training dataset.
        """

        scores, _ = self.neigh.kneighbors(self.x[unique_x], size)
        return -scores.sum(axis=1) / len(scores)

    def _check_densest(self, x_kneighbors, dense_scores, unique_x):
        """Check if the sample x is the densest sample with its neighbors.
        :param x_kneighbors: the array of neighbor indices.
        :param dense_scores: the array of calculated dense scores.
        :param unique_x: the array of unique x indices.
        """

        maximum = dense_scores[np.where(unique_x == x_kneighbors[0])[0]]
        for idx in range(1, self.k):
            if dense_scores[np.where(unique_x == x_kneighbors[idx])[0]] > maximum:
                return False
        return True

    def fit(self, x, y):
        """Run the algorithm and return selected samples.
        :param x: the training set.
        :param y: the labels of the training set.
        """

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
        dense_scores = self._dense(unique_x, size)
        selected = [self._check_densest(x_kneighbors, dense_scores, unique_x) for x_kneighbors in dense_x]
        selected = dense_x[selected]
        selected = selected[:, 0]

        # select from the boundary samples.
        selected_boundary = self.neigh_ind[irrelevance_scores >= int(self.k / 2)]
        selected_boundary = selected_boundary[:, 0]

        return np.concatenate((selected, selected_boundary))
