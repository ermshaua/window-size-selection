import sys, os, shutil

import ruptures.exceptions

sys.path.insert(0, "../")

import numpy as np
np.random.seed(2357)

import pandas as pd
import daproli as dp
import ruptures as rpt

from src.data_loader import load_tssb_dataset
from src.segmentation.floss import floss
from src.segmentation.window import window
from src.segmentation.clasp.segmentation import segmentation

from benchmark.metrics import f_measure, covering
from tqdm import tqdm


def evaluate_candidate(candidate_name, eval_func, wsd, fac=1, columns=None, n_jobs=1, verbose=0, **seg_kwargs):
    df_tssb = load_tssb_dataset()
    df_wsd = pd.read_csv("../experiments/segmentation/window_sizes.csv")

    df_tssb.window_size = np.asarray(df_wsd[wsd] * fac, dtype=np.int64)

    df_eval = dp.map(
        lambda _, args: eval_func(*args, **seg_kwargs),
        tqdm(list(df_tssb.iterrows()), disable=verbose<1),
        ret_type=list,
        verbose=0,
        n_jobs=n_jobs,
    )

    if columns is None:
        columns = ["name", "true_cps", "found_cps", "f1_score", "covering_score"]

    df_eval = pd.DataFrame.from_records(
        df_eval,
        index="name",
        columns=columns,
    )

    print(f"{candidate_name}: mean_f1_score={np.round(df_eval.f1_score.mean(), 3)}, mean_covering_score={np.round(df_eval.covering_score.mean(), 3)}")
    return df_eval


def evaluate_floss_candidate(name, window_size, cps, ts, **seg_kwargs):
    if window_size <= 0:
        return name, cps.tolist(), [], 0., 0., np.zeros(ts.shape[0])

    cac, found_cps = floss(ts, window_size=window_size, return_cac=True, n_cps=len(cps))

    f1_score = f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01))
    covering_score = covering({0: cps}, found_cps, ts.shape[0])

    # print(f"{name}: Found CPs: {found_cps} F1-Score: {np.round(f1_score, 3)}, Covering-Score: {np.round(covering_score, 3)}")
    return name, cps.tolist(), found_cps.tolist(), np.round(f1_score, 3), np.round(covering_score, 3), cac


def evaluate_floss(exp_path, n_jobs, verbose):
    exp_name = "floss"

    if os.path.exists(exp_path + exp_name):
        shutil.rmtree(exp_path + exp_name)

    os.mkdir(exp_path + exp_name)

    df_wsd = pd.read_csv("../experiments/segmentation/window_sizes.csv")

    for algo in df_wsd.columns[2:]:
        candidate_name = f"{exp_name}_{algo}"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            eval_func=evaluate_floss_candidate,
            n_jobs=n_jobs,
            verbose=verbose,
            wsd=algo,
            columns=["name", "true_cps", "found_cps", "f1_score", "covering_score", "cac"]
        )

        df.to_csv(f"{exp_path}{exp_name}/{candidate_name}.csv")


def evaluate_window_candidate(name, window_size, cps, ts, **seg_kwargs):
    if window_size <= 0:
        return name, cps.tolist(), [], 0., 0.

    try:
        found_cps = window(ts, window_size, n_cps=len(cps))
        f1_score = f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01))
        covering_score = covering({0: cps}, found_cps, ts.shape[0])
    except rpt.exceptions.NotEnoughPoints:
        found_cps = np.zeros(0, dtype=np.int64)
        f1_score = 0.
        covering_score = 0.

    # print(f"{name}: Found CPs: {found_cps} F1-Score: {np.round(f1_score, 3)}, Covering-Score: {np.round(covering_score, 3)}")
    return name, cps.tolist(), found_cps.tolist(), np.round(f1_score, 3), np.round(covering_score, 3)


def evaluate_window(exp_path, n_jobs, verbose):
    exp_name = "window"

    if os.path.exists(exp_path + exp_name):
        shutil.rmtree(exp_path + exp_name)

    os.mkdir(exp_path + exp_name)

    df_wsd = pd.read_csv("../experiments/segmentation/window_sizes.csv")

    for algo in df_wsd.columns[2:]:
        candidate_name = f"{exp_name}_{algo}"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            eval_func=evaluate_window_candidate,
            n_jobs=n_jobs,
            verbose=verbose,
            wsd=algo,
        )

        df.to_csv(f"{exp_path}{exp_name}/{candidate_name}.csv")


def evaluate_clasp_candidate(name, window_size, cps, ts):
    if window_size <= 0:
        return name, cps.tolist(), [], 0., 0., np.zeros(ts.shape[0])

    profile, window_size, found_cps, found_scores = segmentation(ts, window_size=window_size, n_change_points=len(cps))

    f1_score = f_measure({0: cps}, found_cps, margin=int(ts.shape[0] * .01))
    covering_score = covering({0: cps}, found_cps, ts.shape[0])

    # print(f"{name}: Window Size: {window_size}, True Change Points: {cps}, Found Change Points: {found_cps}, Scores: {np.round(found_scores, 3)}, F1-Score: {np.round(f1_score, 3)}, Covering-Score: {np.round(covering_score, 3)}")
    return name, cps.tolist(), found_cps.tolist(), np.round(f1_score, 3), np.round(covering_score, 3), profile.tolist()


def evaluate_clasp(exp_path, n_jobs, verbose):
    exp_name = "clasp"

    if os.path.exists(exp_path + exp_name):
        shutil.rmtree(exp_path + exp_name)

    os.mkdir(exp_path + exp_name)

    df_wsd = pd.read_csv("../experiments/segmentation/window_sizes.csv")

    for algo in df_wsd.columns[2:]:
        candidate_name = f"{exp_name}_{algo}"
        print(f"Evaluating parameter candidate: {candidate_name}")

        df = evaluate_candidate(
            candidate_name,
            eval_func=evaluate_clasp_candidate,
            n_jobs=n_jobs,
            verbose=verbose,
            wsd=algo,
            columns=["name", "true_cps", "found_cps", "f1_score", "covering_score", "clasp"]
        )

        df.to_csv(f"{exp_path}{exp_name}/{candidate_name}.csv")


if __name__ == '__main__':
    exp_path = "../experiments/segmentation/"
    n_jobs, verbose = -1, 0

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    # evaluate_floss(exp_path, n_jobs, verbose)
    # evaluate_window(exp_path, n_jobs, verbose)
    evaluate_clasp(exp_path, n_jobs, verbose)