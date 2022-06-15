import numpy as np
import daproli as dp
import pandas as pd

from src.segmentation.clasp.scoring import binary_roc_auc_score
from src.segmentation.clasp.ensemble_clasp import ClaSPEnsemble

from src.segmentation.clasp.penalty import rank_sums_test

from queue import PriorityQueue


def segmentation(time_series, window_size, n_change_points=None, min_seg_size=None, penalty=rank_sums_test, k_neighbours=3, n_iter=30, score=binary_roc_auc_score, offset=.05, **penalty_args):
    queue = PriorityQueue()

    window_size = max(3, window_size)

    if min_seg_size is None:
        min_seg_size = int(max(10 * window_size, offset * time_series.shape[0]))

    if n_change_points is None:
        penalize = True
        n_change_points = time_series.shape[0]
    else:
        penalize = False

    clasp = ClaSPEnsemble(
        window_size=window_size,
        k_neighbours=k_neighbours,
        n_iter=n_iter,
        score=score,
        offset=offset,
        min_seg_size=min_seg_size,
    )

    profile = clasp.fit_transform(time_series)
    change_point = np.argmax(profile)
    global_score = np.max(profile)

    tc_profile, tc_knn, tc_change_point = clasp.applc_tc(profile, change_point)

    if penalize is False or (tc_profile is not None and penalty(tc_profile, tc_knn, tc_change_point, window_size, **penalty_args)):
        queue.put((-global_score, (np.arange(time_series.shape[0]).tolist(), profile, change_point, global_score)))

    change_points = []
    scores = []

    for idx in range(n_change_points):
        # happens if no valid change points exist anymore
        if queue.empty() is True: break

        priority, (local_range, local_profile, change_point, local_score) = queue.get()

        if priority != 0 and _cp_valid(change_point, change_points, time_series.shape[0], min_seg_size):
            change_points.append(change_point)
            scores.append(local_score)

            ind = np.asarray(local_range[:-window_size+1], dtype=np.int64)
            profile[ind] = np.max([profile[ind], local_profile], axis=0)

        if idx == n_change_points-1 or len(change_points) == n_change_points:
            break

        left_range = np.arange(local_range[0], change_point).tolist()
        right_range = np.arange(change_point, local_range[-1]).tolist()

        for local_range in (left_range, right_range):
            _local_segmentation(
                queue,
                time_series,
                window_size,
                clasp.interval_knn,
                profile,
                local_range,
                change_points,
                k_neighbours,
                n_iter,
                score,
                min_seg_size,
                offset,
                penalize,
                penalty,
                penalty_args,
            )

    change_points, scores = np.array(change_points), np.array(scores)

    profile[np.isinf(profile)] = np.nan
    profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()

    return profile, window_size, change_points, scores


def _cp_valid(candidate, change_points, n_timepoints, min_seg_size):
    for change_point in [0] + change_points + [n_timepoints]:
        left_begin = max(0, change_point - min_seg_size)
        right_end = min(n_timepoints, change_point + min_seg_size)

        if candidate in range(left_begin, right_end):
            return False

    return True


def _local_segmentation(queue, time_series, window_size, interval_knn, profile, ts_range, change_points, k_neighbours, n_iter, score, min_seg_size, offset, penalize, penalty, penalty_args):
    if len(ts_range) < 2*min_seg_size: return
    n_timepoints = time_series.shape[0]

    # compute local profile and change point
    local_clasp = ClaSPEnsemble(
        window_size=window_size,
        k_neighbours=k_neighbours,
        n_iter=n_iter,
        score=score,
        interval_knn=interval_knn,
        interval=(ts_range[0], ts_range[-1]+1),
        offset=offset,
        min_seg_size=min_seg_size,
    )

    local_profile = local_clasp.fit_transform(time_series[ts_range])
    local_change_point = np.argmax(local_profile)
    local_score = local_profile[local_change_point]

    global_change_point = ts_range[0] + local_change_point
    global_score = profile[global_change_point]

    # check if change point is a trivial match
    if not _cp_valid(global_change_point, change_points, n_timepoints, min_seg_size):
        return

    tc_profile, tc_knn, tc_change_point = local_clasp.applc_tc(local_profile, local_change_point)

    # apply penalty checks
    if penalize is False or (tc_profile is not None and penalty(tc_profile, tc_knn, tc_change_point, window_size, **penalty_args)):
       queue.put((-local_score, [ts_range, local_profile, global_change_point, local_score]))







