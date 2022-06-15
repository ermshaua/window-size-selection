import numpy as np

from src.utils import sliding_window
from sklearn.ensemble import IsolationForest


def isolation_forest(ts, window_size, test_start):
    W = sliding_window(ts, max(3, window_size))

    est = IsolationForest().fit(W[:test_start-window_size+1,:])
    profile = est.score_samples(W[test_start:,:])

    return test_start + np.argmin(profile)



