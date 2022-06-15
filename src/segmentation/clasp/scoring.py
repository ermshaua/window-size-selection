import numpy as np

from numba import njit


@njit(fastmath=True, cache=True)
def binary_accuracy_score(y_true, y_pred):
    tptn = np.sum(y_true == y_pred)
    tptnfpfn = y_true.shape[0]
    return np.float64(tptn / tptnfpfn)


@njit(fastmath=True, cache=True)
def binary_f1_score(y_true, y_pred):
    f1_scores = np.zeros(shape=2, dtype=np.float64)

    for label in (0,1):
        tp = np.sum(np.logical_and(y_true == label, y_pred == label))
        fp = np.sum(np.logical_and(y_true != label, y_pred == label))
        fn = np.sum(np.logical_and(y_true == label, y_pred != label))

        pr = tp / (tp + fp)
        re = tp / (tp + fn)

        f1 = 2 * (pr * re) / (pr + re)

        f1_scores[label] = f1

    return np.mean(f1_scores)


@njit(fastmath=True, cache=True)
def binary_roc_auc_score(y_score, y_true):
    # make y_true a boolean vector
    y_true = (y_true == 1)

    # sort scores and corresponding truth values (y_true is sorted by design)
    desc_score_indices = np.arange(y_score.shape[0])[::-1]

    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.concatenate((
        distinct_value_indices,
        np.array([y_true.size - 1])
    ))

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.concatenate((np.array([0]), tps))
    fps = np.concatenate((np.array([0]), fps))

    if fps[-1] <= 0 or tps[-1] <= 0:
        return np.nan

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    if fpr.shape[0] < 2:
        return np.nan

    direction = 1
    dx = np.diff(fpr)

    if np.any(dx < 0):
        if np.all(dx <= 0): direction = -1
        else: return np.nan

    area = direction * np.trapz(tpr, fpr)
    return area