import sys, os, shutil

import pandas as pd

sys.path.insert(0, "../")

import numpy as np
np.random.seed(2357)

import daproli as dp


from src.data_loader import load_ucr_anomaly_dataset
from src.window_size.period import dominant_fourier_freq, highest_autocorrelation
from src.window_size.suss import suss
from src.window_size.mwf import mwf
from src.window_size.autoperiod import autoperiod
from src.window_size.robustperiod import robust_period


def evaluate_window_size(exp_path, n_jobs, verbose):
    exp_name = "anomaly_detection"

    if os.path.exists(exp_path + exp_name):
        shutil.rmtree(exp_path + exp_name)

    os.mkdir(exp_path + exp_name)

    algorithms = [
        ("Human", "human"),
        ("FFT", dominant_fourier_freq),
        ("ACF", highest_autocorrelation),
        ("SuSS", suss),
        ("MWF", mwf),
        ("Autoperiod", autoperiod),
        ("RobustPeriod", robust_period)
    ]

    df_ucr = load_ucr_anomaly_dataset(verbose=1, n_jobs=-1)

    df_wsd = pd.DataFrame()
    df_wsd["dataset"] = df_ucr["name"]

    ts = dp.map(lambda test_start, ts: ts[:test_start], zip(df_ucr.test_start, df_ucr.time_series), ret_type=list)

    for candidate_name, algorithm in algorithms:
        print(f"Evaluating window size candidate candidate: {candidate_name}")

        if candidate_name == "Human":
            window_sizes = df_ucr.anomaly_end.to_numpy() - df_ucr.anomaly_start.to_numpy()
        else:
            window_sizes = dp.map(algorithm, ts, expand_args=False, ret_type=np.array, n_jobs=n_jobs, verbose=verbose)

        df_wsd[candidate_name] = window_sizes

    df_wsd.to_csv(f"{exp_path}{exp_name}/window_sizes.csv")


if __name__ == '__main__':
    exp_path = "../experiments/"
    n_jobs, verbose = -1, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_window_size(exp_path, n_jobs, verbose)