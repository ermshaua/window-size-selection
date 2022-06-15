import numpy as np
import pandas as pd


def suss_score(time_series, window_size, stats):
    roll = pd.Series(time_series).rolling(window_size)
    ts_mean, ts_std, ts_min_max = stats

    roll_mean = roll.mean().to_numpy()[window_size:]
    roll_std = roll.std(ddof=0).to_numpy()[window_size:]
    roll_min = roll.min().to_numpy()[window_size:]
    roll_max = roll.max().to_numpy()[window_size:]

    X = np.array([
        roll_mean - ts_mean,
        roll_std - ts_std,
        (roll_max - roll_min) - ts_min_max
    ])

    X = np.sqrt(np.sum(np.square(X), axis=0)) / np.sqrt(window_size)

    return np.mean(X)


def suss(time_series, lbound=10, threshold=.89):
    time_series = (time_series - time_series.min()) / (time_series.max() - time_series.min())

    ts_mean = np.mean(time_series)
    ts_std = np.std(time_series)
    ts_min_max = np.max(time_series) - np.min(time_series)

    stats = (ts_mean, ts_std, ts_min_max)

    max_score = suss_score(time_series, 1, stats)
    min_score = suss_score(time_series, time_series.shape[0]-1, stats)

    exp = 0

    # exponential search (to find window size interval)
    while True:
        window_size = 2 ** exp

        if window_size < lbound:
            exp += 1
            continue

        score = 1 - (suss_score(time_series, window_size, stats) - min_score) / (max_score - min_score)

        if score > threshold:
            break

        exp += 1

    lbound, ubound = max(lbound, 2 ** (exp - 1)), 2 ** exp + 1

    # binary search (to find window size in interval)
    while lbound <= ubound:
        window_size = int((lbound + ubound) / 2)
        score = 1 - (suss_score(time_series, window_size, stats) - min_score) / (max_score - min_score)

        if score < threshold:
            lbound = window_size+1
        elif score > threshold:
            ubound = window_size-1
        else:
            break

    return 2*lbound