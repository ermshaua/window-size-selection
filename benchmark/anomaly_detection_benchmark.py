import sys, os, shutil

sys.path.insert(0, "../")

import numpy as np
np.random.seed(2357)

import pandas as pd
import daproli as dp

from src.data_loader import load_ucr_anomaly_dataset
from src.anomaly_detection.discord import discord
from src.anomaly_detection.isolation_forest import isolation_forest
from src.anomaly_detection.svm import svm

from benchmark.metrics import anomaly_match
from tqdm import tqdm


def evaluate_candidate(candidate_name, eval_func, wsd, fac=1, columns=None, n_jobs=1, verbose=0, **seg_kwargs):
    df_ucr = load_ucr_anomaly_dataset(n_jobs=n_jobs, verbose=verbose)
    df_wsd = pd.read_csv("../experiments/anomaly_detection/window_sizes.csv")

    df_anomaly = pd.DataFrame()
    df_anomaly["name"] = df_ucr["name"]
    df_anomaly["window_size"] = np.asarray(df_wsd[wsd] * fac, dtype=np.int64)
    df_anomaly["test_start"] = df_ucr.test_start
    df_anomaly["anomaly_start"] = df_ucr.anomaly_start
    df_anomaly["anomaly_end"] = df_ucr.anomaly_end
    df_anomaly["time_series"] = df_ucr.time_series

    # length filter
    # df_anomaly = df_anomaly[df_anomaly.time_series.apply(len) < 100_000]

    df_eval = dp.map(
        lambda _, args: eval_func(*args, **seg_kwargs),
        tqdm(list(df_anomaly.iterrows()), disable=verbose<1),
        ret_type=list,
        verbose=0,
        n_jobs=n_jobs,
    )

    if columns is None:
        columns = ["name", "true_start", "true_end", "prediction", "match"]

    df_eval = pd.DataFrame.from_records(
        df_eval,
        index="name",
        columns=columns,
    )

    print(f"{candidate_name}: mean_acc={np.round(np.sum(df_eval.match) / df_eval.shape[0], 3)}")
    return df_eval


def evaluate_discord_candidate(name, window_size, test_start, anomaly_start, anomaly_end, ts):
    if window_size <= 0:
        return name, anomaly_start, anomaly_end, -1, False

    prediction = discord(ts, window_size=window_size, test_start=test_start)
    match = anomaly_match(prediction, anomaly_start, anomaly_end)
    return name, anomaly_start, anomaly_end, prediction, match


def evaluate_discord(exp_path, n_jobs, verbose):
    exp_name = "discord"

    if os.path.exists(exp_path + exp_name):
        shutil.rmtree(exp_path + exp_name)

    os.mkdir(exp_path + exp_name)

    df_wsd = pd.read_csv("../experiments/segmentation/window_sizes.csv")

    for algo in df_wsd.columns[2:]:
        candidate_name = f"{exp_name}_{algo}"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            eval_func=evaluate_discord_candidate,
            n_jobs=n_jobs,
            verbose=verbose,
            wsd=algo,
        )

        df.to_csv(f"{exp_path}{exp_name}/{candidate_name}.csv")


def evaluate_if_candidate(name, window_size, test_start, anomaly_start, anomaly_end, ts):
    if window_size <= 0:
        return name, anomaly_start, anomaly_end, -1, False

    prediction = isolation_forest(ts, window_size=window_size, test_start=test_start)
    match = anomaly_match(prediction, anomaly_start, anomaly_end)
    return name, anomaly_start, anomaly_end, prediction, match


def evaluate_if(exp_path, n_jobs, verbose):
    exp_name = "isolation_forest"

    if os.path.exists(exp_path + exp_name):
        shutil.rmtree(exp_path + exp_name)

    os.mkdir(exp_path + exp_name)

    df_wsd = pd.read_csv("../experiments/segmentation/window_sizes.csv")

    for algo in df_wsd.columns[2:]:
        candidate_name = f"{exp_name}_{algo}"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            eval_func=evaluate_if_candidate,
            n_jobs=n_jobs,
            verbose=verbose,
            wsd=algo,
        )

        df.to_csv(f"{exp_path}{exp_name}/{candidate_name}.csv")


def evaluate_svm_candidate(name, window_size, test_start, anomaly_start, anomaly_end, ts):
    if window_size <= 0:
        return name, anomaly_start, anomaly_end, -1, False

    prediction = svm(ts, window_size=window_size, test_start=test_start)
    match = anomaly_match(prediction, anomaly_start, anomaly_end)
    return name, anomaly_start, anomaly_end, prediction, match


def evaluate_svm(exp_path, n_jobs, verbose):
    exp_name = "svm"

    if os.path.exists(exp_path + exp_name):
        shutil.rmtree(exp_path + exp_name)

    os.mkdir(exp_path + exp_name)

    df_wsd = pd.read_csv("../experiments/segmentation/window_sizes.csv")

    for algo in df_wsd.columns[2:]:
        candidate_name = f"{exp_name}_{algo}"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            eval_func=evaluate_svm_candidate,
            n_jobs=n_jobs,
            verbose=verbose,
            wsd=algo,
        )

        df.to_csv(f"{exp_path}{exp_name}/{candidate_name}.csv")


if __name__ == '__main__':
    exp_path = "../experiments/anomaly_detection/"
    n_jobs, verbose = -1, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    evaluate_discord(exp_path, n_jobs, verbose)
    evaluate_if(exp_path, n_jobs, verbose)
    evaluate_svm(exp_path, n_jobs, verbose)