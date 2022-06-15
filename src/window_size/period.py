import math

import numpy as np
import daproli as dp

from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks


def dominant_fourier_freq(ts, min_size=10, max_size=1000): #
    fourier = np.fft.fft(ts)
    freq = np.fft.fftfreq(ts.shape[0], 1)

    magnitudes = []
    window_sizes = []

    for coef, freq in zip(fourier, freq):
        if coef and freq > 0:
            window_size = int(1 / freq)
            mag = math.sqrt(coef.real * coef.real + coef.imag * coef.imag)

            if window_size >= min_size and window_size < max_size:
                window_sizes.append(window_size)
                magnitudes.append(mag)

    return window_sizes[np.argmax(magnitudes)]


def highest_autocorrelation(ts, min_size=10, max_size=1000):
    acf_values = acf(ts, fft=True, nlags=int(ts.shape[0]/2))

    peaks, _ = find_peaks(acf_values)
    peaks = peaks[np.logical_and(peaks >= min_size, peaks < max_size)]
    corrs = acf_values[peaks]

    if peaks.shape[0] == 0:
        return -1

    return peaks[np.argmax(corrs)]