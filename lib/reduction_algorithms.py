# -*- coding: utf-8 -*-
"""Data Reduction Algorithms.

This module provides implementations for data reduction algorithms (for instance selection).
"""

import os

import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.utils import to_categorical


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

    def select_boundary(self, x, y):

        self.x = x
        self.y = y
        self.neigh.fit(x)
        self.neigh_dist, self.neigh_ind = self.neigh.kneighbors(self.x)
        irrelevance_scores = np.array([self._irrelevance(x_kneighbors) for x_kneighbors in self.neigh_ind])
        selected_boundary = self.neigh_ind[irrelevance_scores >= int(self.k / 2)]
        selected_boundary = selected_boundary[:, 0]

        return selected_boundary, irrelevance_scores

    def fit(self, x, y):
        """Run the algorithm and return selected samples.
        :param x: the training set.
        :param y: the labels of the training set.
        """

        # start the EGDIS algorithm
        # select from the boundary samples.
        selected_boundary, irrelevance_scores = self.select_boundary(x, y)

        # calculate the irrelevance of each sample, 0 means no neighbors have different labels.
        dense_x = self.neigh_ind[irrelevance_scores == 0]
        unique_x = np.unique(dense_x)

        # calculate the dense scores for unique and irrelevance_scores == 0 samples.
        size = len(self.x)
        dense_scores = self._dense(unique_x, size)
        selected = [self._check_densest(x_kneighbors, dense_scores, unique_x) for x_kneighbors in dense_x]
        selected = dense_x[selected]
        selected = selected[:, 0]

        return np.concatenate((selected, selected_boundary))


def _group_consecutives(attribute_with_label, step=0):
    """Return list of consecutive lists of numbers from vals (number list)."""


    run = []
    result = [run]
    expect_value = None
    expect_label = None
    for v in attribute_with_label:
        if (v[1] == expect_label) or (expect_label is None) or (
                expect_value is not None and (np.isclose(v[0], expect_value))):
            run.append(v[0])
            if v[0] == expect_value:
                print("Find consecutive attributes, implement _resort is necessary.")
        else:
            run = [v[0]]
            result.append(run)
        expect_label = v[1] + step
        expect_value = v[0]
    return result


def _resort(sort_index, attributes, labels):
    """Resort the orders within consecutive repeating attribute values"""
    raise NotImplementedError("This function is not implemented because no consecutive repeating attributes found for "
                              "now.")


class POP(object):
    """Implementation of the patterns by ordered projections(POP) algorithm. Cited: Riquelme J C, Aguilar-Ruiz J S,
    Toro M. Finding representative patterns with ordered projections[J]. Pattern Recognition, 2003, 36(4): 1009-1018.
    """

    def __init__(self):
        """Init a new POP instance."""

        self.x = None
        self.y = None

    def _cal_weakness(self, attribute):
        """Calculate the weakness of the instances for one attribute.
        :param attribute: an 1D array of a single attribute across all samples.
        """

        weakness = np.zeros(len(self.x))
        assert len(attribute) == len(self.y), "The length doesn't match."
        # sort the samples according to the attribute value.
        sort_index = np.argsort(attribute)

        attribute_with_label = np.concatenate((sort_index.reshape(-1, 1), self.y[sort_index].reshape(-1, 1)), axis=1)
        grouped_index = _group_consecutives(attribute_with_label)
        for group in grouped_index:
            if len(group) >= 3:
                weakness[group[1:-1]] += 1

        return weakness

    def fit(self, x, y):
        """Get the weakness score for each sample.
        :param x: the training set.
        :param y: the the label of the samples.
        """

        self.x = x
        if len(y.shape) > 1:
            self.y = np.argmax(y, axis=1)
        else:
            self.y = y

        weakness_list = np.array([self._cal_weakness(attribute) for attribute in self.x.T]).sum(axis=0)

        return weakness_list


def rank_data_according_to_score(train_scores, reverse=False, random=False):
    """Rank the scores in descending order."""
    rank = np.asarray(sorted(range(len(train_scores)), key=lambda k: train_scores[k], reverse=True))
    # if reverse:
    #     res = np.flip(res, 0)
    # if random:
    #     np.random.shuffle(res)
    return rank


class CL(object):
    """Implementation of the Curriculum Learning. Cited: Hacohen G, Weinshall D. On the power of curriculum
        learning in training deep networks[J]. arXiv preprint arXiv:1904.03626, 2019."""

    def __init__(self):
        """Init a CL instance"""
        self.classifier_layer = None

    def fit_dataset(self, classes=None, dataset=None, clf=None):
        """Fit the dataset information.
        :param clf: trained keras classifier.
        :param classes: the number of classes.
        :param dataset: the name string of the dataset.
        """
        if clf:
            self.classifier_layer = clf
        else:
            self.classifier_layer = tf.keras.Sequential([
                tf.keras.layers.Dense(classes,  # output dim is one score per each class
                                      activation='softmax',
                                      kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.1),
                                      )
            ])
            self.classifier_layer.build([None, 128])
            self.classifier_layer.load_weights(os.path.join(os.getcwd(), "models", dataset, "classifier.h5"))

    def fit(self, x, y):
        """Evaluate all the training samples and rank them based on scores.
        :param x: the training set.
        :param y: the the label of the samples.
        """

        scores = self.classifier_layer.predict_proba(x)
        assert scores.shape == y.shape, "The shapes don't match, {} with {}".format(scores.shape, y.shape)
        scores = scores * y
        scores = scores.sum(axis=1)
        rank = rank_data_according_to_score(scores)

        return rank, scores


class WCL(object):
    """Weighted Curriculum Learning. Use the weakness calculated by POP to weight the CL score, original work. Combine
    difficulty with weakness. If weakness is 0, and the score is small, then the score is enlarged. If weakness is large
    , then the score is scaled to a smaller value."""

    def __init__(self, k=4):
        """Init a wcl instance.
        :param k: the nearest neighbor parameter, should be k+1 if the instance is in the dataset.
        """

        self.cl = CL()
        self.egdis = EGDIS(k=k)

    def fit_dataset(self, classes=None, dataset=None, clf=None):
        """Fit the dataset information.
        :param classes: the number of classes.
        :param dataset: the name string of the dataset.
        """

        if clf:
            self.cl.fit_dataset(clf=clf)
        else:
            self.cl.fit_dataset(classes, dataset)

    def fit(self, x, y, classes, mode=1):
        """Weight the CL scores with weakness.
        :param mode: which weight mode to use.
        :param x: the training set.
        :param y: the the label of the samples.
        """

        selected_boundary_idx, _ = self.egdis.select_boundary(x, y)
        _, scores = self.cl.fit(x, to_categorical(y, num_classes=classes))
        # # attempt 1: scale down the non-boundary samples with 1/e^(weakness/128).
        # if mode == 1:
        #     weighted_scores = scores * 1 / np.exp(weakness / 128)
        #
        # attempt 2: scale up the boundary samples, keep all boundary samples.
        # if mode == 2:
        #     weighted_scores = scores.copy()
        #     weighted_scores[weakness == 0] = 1
        #
        # # attempt 3: hybrid method.
        # if mode == 3:
        #     weighted_scores = scores * 1 / np.exp(weakness / 128)
        #     weighted_scores[weakness == 0] = 1

        # rank = rank_data_according_to_score(weighted_scores)

        # return rank, weighted_scores

        return scores, selected_boundary_idx
