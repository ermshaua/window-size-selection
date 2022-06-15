import sys, os, shutil

import pandas as pd

sys.path.insert(0, "../")

import numpy as np
np.random.seed(2357)

import daproli as dp


from src.data_loader import load_tssb_dataset
from src.window_size.period import dominant_fourier_freq, highest_autocorrelation
from src.window_size.suss import suss
from src.window_size.mwf import mwf
from src.window_size.autoperiod import autoperiod
from src.window_size.robustperiod import robust_period


if __name__ == '__main__':
    selection = 1
    df_tssb = load_tssb_dataset()

    name, w, cps, ts = df_tssb.iloc[selection,:]
    robust_period(ts)