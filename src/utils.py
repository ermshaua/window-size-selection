import numpy as np


def sliding_window(time_series, window):
    shape = time_series.shape[:-1] + (time_series.shape[-1] - window + 1, window)
    strides = time_series.strides + (time_series.strides[-1],)
    return np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)