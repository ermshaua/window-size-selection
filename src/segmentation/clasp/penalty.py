import numpy as np

from src.segmentation.clasp.clasp import _labels
from scipy.stats import ranksums


def _gini(knn, change_point, window_size):
    _, y_pred = _labels(knn, change_point, window_size)

    gini = 0
    for label in (0,1): gini += np.square(np.sum(y_pred == label) / y_pred.shape[0])
    return 1 - gini


def _gini_part(knn, change_point, window_size):
    _, y_pred = _labels(knn, change_point, window_size)

    left_y_pred = y_pred[:change_point]
    left_gini = 0
    for label in (0,1): left_gini += np.square(np.sum(left_y_pred == label) / left_y_pred.shape[0])
    left_gini = 1 - left_gini

    right_y_pred = y_pred[change_point:]
    right_gini = 0
    for label in (0, 1): right_gini += np.square(np.sum(right_y_pred == label) / right_y_pred.shape[0])
    right_gini = 1 - right_gini

    return (left_y_pred.shape[0] / y_pred.shape[0]) * left_gini + (right_y_pred.shape[0] / y_pred.shape[0]) * right_gini


def gini_gain_test(profile, knn, change_point, window_size, threshold=.02):
    gini = _gini(knn, change_point, window_size) - _gini_part(knn, change_point, window_size)
    return gini >= threshold


def rank_sums_test(profile, knn, change_point, window_size, threshold=1e-15):
    _, y_pred = _labels(knn, change_point, window_size)
    _, p = ranksums(y_pred[:change_point], y_pred[change_point:])
    return p <= threshold


def threshold_test(profile, knn, change_point, window_size, min_score=.75):
    return profile[change_point] >= min_score