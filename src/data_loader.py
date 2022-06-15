import os, json

import numpy as np
import pandas as pd
import daproli as dp

from tqdm import tqdm
from ast import literal_eval
from os.path import exists
from scipy.stats import zscore


def load_tssb_dataset():
    desc_filename = "../datasets/TSSB/desc.txt"
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines(): desc_file.append(line.split(","))

    df = []

    for row in desc_file:
        (ts_name, window_size), change_points = row[:2], row[2:]

        ts = np.loadtxt(fname=os.path.join('../datasets/TSSB/', ts_name + '.txt'),
                        dtype=np.float64)
        df.append(
            (ts_name, int(window_size), np.array([int(_) for _ in change_points]), ts))

    return pd.DataFrame.from_records(df,
                                     columns=["name", "window_size", "change_points",
                                              "time_series"])


def load_ucr_anomaly_dataset(selection=None, verbose=0, n_jobs=1):
    file_names = os.listdir("../datasets/UCRAnomaly/")
    file_names = dp.filter(lambda fn: fn.endswith(".txt"), file_names)

    if selection is None:
        selection = slice(len(file_names))

    if isinstance(selection, int):
        selection = slice(selection, selection + 1)

    def read_ts(file_name):
        file_name, _ = os.path.splitext(file_name)
        file_name = file_name.split("_")
        name_id, name, test_start, anomaly_start, anomaly_end = file_name[0], file_name[
            3], file_name[4], file_name[5], file_name[6]
        ts = np.loadtxt(fname=os.path.join(
            f'../datasets/UCRAnomaly/{name_id}_UCR_Anomaly_{name}_{test_start}_{anomaly_start}_{anomaly_end}.txt'),
                        dtype=np.float64).flatten()

        return int(name_id), name, int(test_start), int(anomaly_start), int(
            anomaly_end), ts

    df = dp.map(read_ts,
                tqdm(np.array(sorted(file_names, key=lambda s: int(s[:3])))[selection],
                     disable=verbose < 1), ret_type=list, n_jobs=n_jobs)
    return pd.DataFrame.from_records(df, columns=["idx", "name", "test_start",
                                                  "anomaly_start", "anomaly_end",
                                                  "time_series"])


##################
####  Motifs #####
##################
def load_motifs_datasets(sampling_factor=10000):
    def resample(data, sampling_factor=10000):
        if len(data) > sampling_factor:
            factor = np.int32(len(data) / sampling_factor)
            data = data[::factor]
        return data

    def read_ground_truth(dataset):
        file = '../datasets/motifs/' + dataset + "_gt.csv"
        if (exists(file)):
            print(file)
            series = pd.read_csv(
                file, index_col=0,
                converters={1: literal_eval, 2: literal_eval, 3: literal_eval})
            return series
        return None

    full_path = '../datasets/motifs/'
    desc_filename = full_path+"desc.csv"
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines(): desc_file.append(line.split(","))

    df = []
    for (ts_name, window_size) in desc_file:
        data = pd.read_csv(full_path + ts_name + ".csv", index_col=0).squeeze("columns")
        print("Dataset Original Length n: ", len(data))

        data = resample(data, sampling_factor)
        print("Dataset Sampled Length n: ", len(data))

        data[:] = zscore(data)
        gt = read_ground_truth(ts_name)
        df.append((ts_name, data.values, window_size, gt))

    return pd.DataFrame.from_records(df, columns=["name", "time_series", "window_size", "gt"])
