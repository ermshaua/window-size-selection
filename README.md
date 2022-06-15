# Window Size Selection
This is the supporting website for the paper "Window Size Selection In Unsupervised Time Series Analytics: A Review and Benchmark". It contains the used source codes, data sets, results and analysis notebooks.

Time series (TS) are sequences of values ordered in time and have in common, that important insights from the data can be drawn by inspecting local substructures, or windows. As such, many state-of-the-art time series data mining (TSDM) methods characterize TS by inspecting local substructures. The window size for extracting such subsequences is a crucial hyper-parameter, and setting an inappropriate value results in poor TSDM results. We provide, for the first time, a systematic survey and experimental study of 6 TS window size selection (WSS) algorithms on three diverse TSDM tasks, namely anomaly detection, segmentation and motif discovery, using state-of-the art TSDM algorithms and benchmarks.

This repository is structured as follows: 

- `benchmark` contains the source codes used for creating the window sizes (for all data sets) as well as the benchmark performances (for all tested algorithms).
- `datasets` consists of the (links to the) time series data used in the study. It can be loaded with the data loader in `src/data_loader.py`.
- `experiments` contains the computed window sizes (for the data sets) as well as the performance results (for the tested algorithms). It can be reproduced by running the scripts in the `benchmark` folder.
- `nootebooks` consists of analysis Jupyter notebooks, used to evaluate the experimental data in the `experiments` folder.
- `src` contains the sources codes for window size selection and the time series analytics.

