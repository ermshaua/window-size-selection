import numpy as np

from src.utils import sliding_window
from sklearn.linear_model import SGDOneClassSVM


def svm(ts, window_size, test_start):
    W = sliding_window(ts, max(3, window_size))

    est = SGDOneClassSVM().fit(W[:test_start-window_size+1,:])
    profile = est.score_samples(W[test_start:,:])

    return test_start + np.argmin(profile)



