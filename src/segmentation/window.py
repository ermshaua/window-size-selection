import numpy as np

import ruptures as rpt


def window(ts, window_size, cost_func="mahalanobis", n_cps=None, offset=0.05):
    transformer = rpt.Window(width=max(window_size, 3), model=cost_func, min_size=int(ts.shape[0] * offset)).fit(ts)
    return np.array(transformer.predict(n_bkps=n_cps)[:-1], dtype=np.int64)

