import numpy as np

import stumpy


def discord(ts, window_size, test_start):
    mp = stumpy.stump(ts, m=max(3, window_size))
    return test_start + np.argsort(mp[test_start:, 0])[-1]



