# Window Size Selection
This is the supporting website for the paper "Window Size Selection In Unsupervised Time Series Analytics: A Review and Benchmark". It contains the used source codes, data sets, results and analysis notebooks.

Time series (TS) are sequences of values ordered in time and have in common, that important insights from the data can be drawn by inspecting local substructures, or windows. As such, many state-of-the-art time series data mining (TSDM) methods characterize TS by inspecting local substructures. The window size for extracting such subsequences is a crucial hyper-parameter, and setting an inappropriate value results in poor TSDM results. We provide, for the first time, a systematic survey and experimental study of 6 TS window size selection (WSS) algorithms on three diverse TSDM tasks, namely anomaly detection, segmentation and motif discovery, using state-of-the art TSDM algorithms and benchmarks.

This repository is structured as follows: 

- `benchmark` contains the source codes used for creating the window sizes (for all data sets) as well as the benchmark performances (for all tested algorithms).
- `datasets` consists of the (links to the) time series data used in the study. It can be loaded with the data loader in `src/data_loader.py`.
- `experiments` contains the computed window sizes (for the data sets) as well as the performance results (for the tested algorithms). It can be reproduced by running the scripts in the `benchmark` folder.
- `notebooks` consists of analysis Jupyter notebooks, used to evaluate the experimental data in the `experiments` folder.
- `src` contains the sources codes for window size selection and the time series analytics.

## Installation

You can download this repository (clicking the download button in the upper right corner). As this repository is a supporting website and not an updated library, we do not recommend to install it, but rather extract and adapt code snippets of interest.

## Citation

The associated paper "Window Size Selection In Unsupervised Time Series Analytics: A Review and Benchmark" is currently under double-blinded review. Thereafter, we provide citation details.

## Resources

The review and benchmark was realized with the code and data sets from multiple authors and projects. We list here the resources we used (and adapted) for our study:

- Anomaly Detection
  - One-Class SVM, Isolation Forest (https://scikit-learn.org/)
  - Discord Discovery (https://stumpy.readthedocs.io/)
- Segmentation
  - Paramter-free ClaSP (https://sites.google.com/view/ts-parameter-free-clasp/)
  - FLOSS (https://stumpy.readthedocs.io/)
  - Window-based Segmentation (https://centre-borelli.github.io/ruptures-docs/)
- Motif Discovery
  - EMMA (https://github.com/jMotif/SAX)
  - Set Finder (https://sites.google.com/site/timeseriesmotifsets/)
- Window Size Detection
  - FFT, ACF (https://numpy.org/)
  - RobustPeriod (https://github.com/ariaghora/robust-period)
  - AutoPeriod (https://github.com/akofke/autoperiod)
  - Multi-Window-Finder (https://sites.google.com/view/multi-window-finder/)
  - SuSS (https://sites.google.com/view/ts-parameter-free-clasp/)