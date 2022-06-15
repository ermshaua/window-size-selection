import numpy as np
import pandas as pd

from src.segmentation.clasp.knn import iterative_knn
from src.segmentation.clasp.scoring import binary_roc_auc_score
from numba import njit, prange


@njit(fastmath=True, cache=True)
def _labels(knn_mask, split_idx, window_size):
    n_timepoints, k_neighbours = knn_mask.shape

    y_true = np.concatenate((
        np.zeros(split_idx, dtype=np.int64),
        np.ones(n_timepoints - split_idx, dtype=np.int64),
    ))

    knn_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int64)

    for i_neighbor in range(k_neighbours):
        neighbours = knn_mask[:, i_neighbor]
        knn_labels[i_neighbor] = y_true[neighbours]

    ones = np.sum(knn_labels, axis=0)
    zeros = k_neighbours - ones
    y_pred = np.asarray(ones > zeros, dtype=np.int64)

    exclusion_zone = np.arange(split_idx - window_size, split_idx)
    y_pred[exclusion_zone] = 1

    return y_true, y_pred


@njit(fastmath=True, cache=False)
def _profile(window_size, knn, score, offset):
    n_timepoints, _ = knn.shape
    profile = np.full(shape=n_timepoints, fill_value=-np.inf, dtype=np.float64)
    offset = max(10*window_size, offset)

    for split_idx in range(offset, n_timepoints - offset):
        y_true, y_pred = _labels(knn, split_idx, window_size)

        try:
            _score = score(y_true, y_pred)

            if not np.isnan(_score):
                profile[split_idx] = _score
        except:
            pass

    return profile


class ClaSP:

    def __init__(self, window_size, k_neighbours=3, score=binary_roc_auc_score, offset=.05):
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.score = score
        self.offset = offset

    def fit(self, time_series):
        return self

    def transform(self, time_series, interpolate=False):
        self.dist, self.knn = iterative_knn(time_series, self.window_size, self.k_neighbours)

        n_timepoints = self.knn.shape[0]
        offset = np.int64(n_timepoints * self.offset)

        profile = _profile(self.window_size, self.knn, self.score, offset)

        if interpolate:
            profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()

        return profile

    def fit_transform(self, time_series, interpolate=False):
        return self.fit(time_series).transform(time_series, interpolate=interpolate)