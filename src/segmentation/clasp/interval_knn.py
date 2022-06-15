import numpy as np
import numpy.fft as fft
from numba import njit


@njit(fastmath=True, cache=True)
def _sliding_mean_std(time_series, window_size):
    # s = np.insert(np.cumsum(time_series), 0, 0)
    # sSq = np.insert(np.cumsum(time_series ** 2), 0, 0)

    s = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(time_series)))
    sSq = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(time_series ** 2)))

    segSum = s[window_size:] - s[:-window_size]
    segSumSq = sSq[window_size:] - sSq[:-window_size]

    movmean = segSum / window_size
    movstd = np.sqrt(1e-9 + segSumSq / window_size - (segSum / window_size) ** 2)

    return [movmean, movstd]


def _sliding_dot(query, time_series):
    m = len(query)
    n = len(time_series)

    time_series_add = 0
    if n % 2 == 1:
        time_series = np.insert(time_series, 0, 0)
        time_series_add = 1

    q_add = 0
    if m % 2 == 1:
        query = np.insert(query, 0, 0)
        q_add = 1

    query = query[::-1]
    query = np.pad(query, (0, n - m + time_series_add - q_add), 'constant')
    trim = m - 1 + time_series_add
    dot_product = fft.irfft(fft.rfft(time_series) * fft.rfft(query))
    return dot_product[trim:]


@njit(fastmath=True, cache=True)
def find_knn(idx, k_neighbours, lbound, ubound, exclusion_radius=0):
    knns = list()

    for neigh_a in idx:
        if len(knns) == k_neighbours: break
        if neigh_a < lbound or neigh_a >= ubound: continue

        valid = True

        for neigh_b in knns:
            if np.abs(neigh_a - neigh_b) <= exclusion_radius:
                valid = False
                break

        if valid is True:
            knns.append(neigh_a)

    return np.array(knns)


@njit(fastmath=True, cache=True)
def argkmin(dist, k):
    args = np.zeros(shape=k, dtype=np.int64)
    vals = np.zeros(shape=k, dtype=np.float64)

    for idx in range(k):
        min_arg = np.nan
        min_val = np.inf

        for kdx, val in enumerate(dist):
            if val < min_val:
                min_val = val
                min_arg = kdx

        min_arg = np.int64(min_arg)

        args[idx] = min_arg
        vals[idx] = min_val

        dist[min_arg] = np.inf

    dist[args] = vals
    return args


@njit(fastmath=True, cache=True)
def _iterative_knn(time_series, window_size, k_neighbours, tcs, dot_first):
    l = len(time_series) - window_size + 1
    exclusion_radius = np.int64(window_size / 2)

    knns = np.zeros(shape=(l, len(tcs) * k_neighbours), dtype=np.int64)
    dists = np.zeros(shape=(l, len(tcs) * k_neighbours), dtype=np.float64)

    dot_prev = None
    means, stds = _sliding_mean_std(time_series, window_size)

    for order in range(0, l):
        if order == 0:
            dot_rolled = dot_first
        else:
            dot_rolled = np.roll(dot_prev, 1) + time_series[order + window_size - 1] * time_series[window_size - 1:l + window_size] - time_series[order - 1] * np.roll(time_series[:l], 1)
            dot_rolled[0] = dot_first[order]

        x_mean = means[order]
        x_std = stds[order]

        # dist is squared
        dist = 2 * window_size * (1 - (dot_rolled - window_size * means * x_mean) / (window_size * stds * x_std))

        # self-join: exclusion zone
        trivialMatchRange = (
            int(max(0, order - np.round(exclusion_radius, 0))),
            int(min(order + np.round(exclusion_radius + 1, 0), l))
        )

        dist[trivialMatchRange[0]:trivialMatchRange[1]] = np.max(dist)
        # idx = np.argsort(dist, kind="mergesort")  #  #

        for kdx, (lbound, ubound) in enumerate(tcs):
            if order < lbound or order >= ubound: continue

            tc_nn = lbound + argkmin(dist[lbound:ubound], k_neighbours)
            # tc_nn = find_knn(idx, k_neighbours, lbound, ubound, exclusion_radius=0)

            knns[order, kdx * k_neighbours:(kdx + 1) * k_neighbours] = tc_nn
            dists[order, kdx * k_neighbours:(kdx + 1) * k_neighbours] = dist[tc_nn]

        dot_prev = dot_rolled

    return dists, knns


class IntervalKneighbours:

    def __init__(self, time_series, window_size, k_neighbours):
        self.time_series = time_series
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.knns = [set() for _ in range(time_series.shape[0])]

    def knn(self, interval, tcs):
        interval_start, interval_end = interval
        time_series = self.time_series[interval_start:interval_end]

        dot_first = _sliding_dot(time_series[:self.window_size], time_series)
        local_dists, local_knns = _iterative_knn(time_series, self.window_size, self.k_neighbours, tcs, dot_first)

        return local_dists, local_knns