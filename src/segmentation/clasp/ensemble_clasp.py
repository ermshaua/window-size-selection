import numpy as np
import pandas as pd
import daproli as dp

from src.segmentation.clasp.interval_knn import IntervalKneighbours
from src.segmentation.clasp.scoring import binary_roc_auc_score
from src.segmentation.clasp.clasp import _profile


class ClaSPEnsemble:

    def __init__(self, window_size, n_iter=30, k_neighbours=3, score=binary_roc_auc_score, min_seg_size=None, interval_knn=None, interval=None, offset=.05, random_state=1379):
        self.window_size = window_size
        self.n_iter = n_iter
        self.k_neighbours = k_neighbours
        self.score = score
        self.min_seg_size = min_seg_size
        self.interval_knn = interval_knn
        self.interval = interval
        self.offset = offset
        self.random_state = random_state

    def fit(self, time_series):
        return self

    def _calculate_tcs(self, time_series):
        tcs = [(0, time_series.shape[0])]
        np.random.seed(self.random_state)

        while len(tcs) < self.n_iter and time_series.shape[0] > 3 * self.min_seg_size:
            lbound, area = np.random.choice(time_series.shape[0], 2, replace=True)

            if time_series.shape[0] - lbound < area:
                area = time_series.shape[0] - lbound

            ubound = lbound + area
            if ubound - lbound < 2 * self.min_seg_size: continue
            tcs.append((lbound, ubound))

        return np.asarray(tcs, dtype=np.int64)

    def _ensemble_profiles(self, time_series):
        self.tcs = self._calculate_tcs(time_series)
        _, self.knn = self.interval_knn.knn(self.interval, self.tcs)

        n_timepoints = self.knn.shape[0]
        offset = np.int64(n_timepoints * self.offset)

        profile = np.full(shape=n_timepoints, fill_value=-np.inf, dtype=np.float64)  #
        bounds = np.full(shape=(n_timepoints, 3), fill_value=-1, dtype=np.int64)

        for idx, (lbound, ubound) in enumerate(self.tcs):
            tc_knn = self.knn[lbound:ubound, idx * self.k_neighbours:(idx + 1) * self.k_neighbours] - lbound

            tc_profile = _profile(self.window_size, tc_knn, self.score, offset)
            not_ninf = np.logical_not(tc_profile == -np.inf)

            tc = (ubound - lbound) / self.knn.shape[0]
            tc_profile[not_ninf] = (2 * tc_profile[not_ninf] + tc) / 3

            change_idx = profile[lbound:ubound] < tc_profile
            change_mask = np.logical_and(change_idx, not_ninf)

            profile[lbound:ubound][change_mask] = tc_profile[change_mask]
            bounds[lbound:ubound][change_mask] = np.array([idx, lbound, ubound])

        return profile, bounds

    def transform(self, time_series, interpolate=False):
        if self.min_seg_size is None:
            self.min_seg_size = int(max(10 * self.window_size, self.offset * time_series.shape[0]))

        if self.interval_knn is None:
            self.interval_knn = IntervalKneighbours(time_series, self.window_size, self.k_neighbours)

        if self.interval is None:
            self.interval = (0, time_series.shape[0])

        profile, self.bounds = self._ensemble_profiles(time_series)

        if interpolate:
            profile[np.isinf(profile)] = np.nan
            profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()

        return profile

    def fit_transform(self, time_series, interpolate=False):
        return self.fit(time_series).transform(time_series, interpolate=interpolate)

    def applc_tc(self, profile, change_point):
        idx, lbound, ubound = self.bounds[change_point]

        if lbound == -1:
            return None, None, None

        tc_profile = profile[lbound:ubound]
        tc_knn = self.knn[lbound:ubound, idx * self.k_neighbours:(idx + 1) * self.k_neighbours] - lbound
        tc_change_point = change_point - lbound

        return tc_profile, tc_knn, tc_change_point