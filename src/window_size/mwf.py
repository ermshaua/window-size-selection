import numpy as np


def movmean(ts, w):
    """
    # faster solution of moving ave
    moving_avg = np.cumsum(ts, dtype=float)
    moving_avg[w:] = moving_avg[w:] - moving_avg[:-w]
    return moving_avg[w-1:] / w
    """
    moving_avg = np.cumsum(ts, dtype=float)
    moving_avg[w:] = moving_avg[w:] - moving_avg[:-w]
    return moving_avg[w - 1:] / w


def mwf(ts, lbound=10, ubound=1_000):
    """
    finidng appropriate window size using movemean
    """
    all_averages = []
    window_sizes = []

    for w in range(lbound, ubound, 1):
        movingAvg = np.array(movmean(ts, w))
        all_averages.append(movingAvg)
        window_sizes.append(w)

    movingAvgResiduals = []

    for i, w in enumerate(window_sizes):
        moving_avg = all_averages[i][:len(all_averages[-1])]
        movingAvgResidual = np.log(abs(moving_avg - (moving_avg).mean()).sum())
        movingAvgResiduals.append(movingAvgResidual)

    b = (np.diff(np.sign(np.diff(movingAvgResiduals))) > 0).nonzero()[0] + 1  # local min

    if len(b) == 0: return -1
    if len(b) < 3: return window_sizes[b[0]]

    reswin = np.array([window_sizes[b[i]] / (i + 1) for i in range(3)])
    w = np.mean(reswin)

    # w = 0.8 * reswin[0] + 0.15 * reswin[1] + 0.05 * reswin[2]
    # conf = np.std(reswin) / np.sqrt(3)

    return int(w)