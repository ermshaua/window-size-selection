import sys, os, shutil

import pandas as pd

sys.path.insert(0, "../")

import numpy as np
np.random.seed(2357)

import daproli as dp


from src.data_loader import load_motifs_datasets
from src.window_size.period import dominant_fourier_freq, highest_autocorrelation
from src.window_size.suss import suss
from src.window_size.mwf import mwf
from src.window_size.autoperiod import autoperiod
from src.window_size.robustperiod import robust_period


def evaluate_window_size(exp_path, n_jobs, verbose):
    exp_name = "motifs"

    if os.path.exists(exp_path + exp_name):
        shutil.rmtree(exp_path + exp_name)

    os.mkdir(exp_path + exp_name)

    algorithms = [
        ("Ground Truth", "GroundTruth"),
        ("FFT", dominant_fourier_freq),
        ("ACF", highest_autocorrelation),
        ("SuSS", suss),
        ("MWF", mwf),
        ("Autoperiod", autoperiod),
        ("RobustPeriod", robust_period)
    ]

    df_motifs = load_motifs_datasets()

    df_wsd = pd.DataFrame()
    df_wsd["dataset"] = df_motifs["name"]

    for candidate_name, algorithm in algorithms:
        print(f"Evaluating window size candidate candidate: {candidate_name}")

        if candidate_name == "Ground Truth":
            window_sizes = df_motifs.window_size.astype(int)
        else:
            window_sizes = dp.map(algorithm, df_motifs.time_series, expand_args=False, ret_type=np.array, n_jobs=n_jobs, verbose=verbose)

        df_wsd[candidate_name] = window_sizes

    df_wsd.to_csv(f"{exp_path}{exp_name}/window_sizes.csv")


if __name__ == '__main__':
    exp_path = "../experiments/"
    n_jobs, verbose = -1, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_window_size(exp_path, n_jobs, verbose)