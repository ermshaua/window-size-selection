import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import biweight_midvariance
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import find_peaks

from .modwt import modwt
from .utils import sinewave, triangle
from .mperioreg_fallback import m_perio_reg
from .huberacf import huber_acf, get_ACF_period
from .fisher import fisher_g_test


def extract_trend(y, reg):
    _, trend = hpfilter(y, reg)
    y_hat = y - trend
    return trend, y_hat


def huber_func(x, c):
    return np.sign(x) * np.minimum(np.abs(x), c)


def MAD(x):
    return np.mean(np.abs(x - np.median(x)))


def residual_autocov(x, c):
    '''
    The \Psi transformation function
    '''
    mu = np.median(x)
    s = MAD(x)
    return huber_func((x - mu)/s, c)


def robust_period_full(x, wavelet_method, num_wavelet, lmb, c, zeta=1.345):
    '''
    Params:
    - x: input signal with shape of (m, n), m is the number of observation and
         n is the number of series
    - wavelet_method:
    - num_wavelet:
    - lmb: Lambda (regularization param) in Hodrick–Prescott (HP) filter
    - c: Huber function hyperparameter
    - zeta: M-Periodogram hyperparameter

    Returns:
    - Array of periods
    - Wavelets
    - bivar
    - Periodograms
    - pval
    - ACF
    '''

    assert wavelet_method.startswith('db'), \
        'wavelet method must be Daubechies family, e.g., db1, ..., db34'

    # 1) Preprocessing
    # ----------------
    # Extract trend and then deterend input series. Then perform residual
    # autocorrelation to remove extreme outliers.
    trend, y_hat = extract_trend(x, lmb)
    y_prime = residual_autocov(y_hat, c)

    # 2) Decoupling multiple periodicities
    # ------------------------------------
    # Perform MODWT and ranking by robust wavelet variance
    W = modwt(y_prime, wavelet_method, level=num_wavelet)

    # compute wavelet variance for all levels
    # TODO Clarifying Lj, so we can omit first Lj from wj
    bivar = np.array([biweight_midvariance(w) for w in W])

    # 3) Robust single periodicity detection
    # --------------------------------------
    # Compute Huber periodogram
    X = np.hstack([W, np.zeros_like(W)])

    periodograms = []
    p_vals = []
    for i, x in enumerate(X):
        # print(f'Calculating periodogram for level {i+1}')
        perio = m_perio_reg(x)

        try:
            p_val, _ = fisher_g_test(perio)
            periodograms.append(perio)
            p_vals.append(p_val)
        except FloatingPointError:
            pass

    periodograms = np.array(periodograms)
    # np.savetxt('periodograms.csv', periodograms, delimiter=',')

    # Compute Huber ACF
    ACF = np.array([huber_acf(p) for p in periodograms])

    periods = []
    for acf in ACF:
        peaks, _ = find_peaks(acf)
        distances = np.diff(peaks)
        final_period = np.median(distances)
        periods.append(final_period)
    periods = np.array(periods)

    periods = []
    for p in periodograms:
        _, final_period, _ = get_ACF_period(p)
        periods.append(final_period)
    periods = np.array(periods)
    final_periods = np.unique(periods[periods > 0])

    return (
        final_periods,  # Periods
        W,              # Wavelets
        bivar,          # bivar
        periodograms,   # periodograms
        p_vals,         # pval
        ACF             # ACF
    )


def robust_period(ts, cutoff=10_000):
    # this implementation takes a long time ...
    if cutoff is not None:
        ts = ts[:cutoff]

    try:
        periods, _, _, _, p_vals, _ = robust_period_full(ts, 'db10', num_wavelet=8, lmb=1e+6, c=2)
        window_size = np.int64(periods[np.argmin(p_vals)])
        return window_size
    except:
        print(f"Could not determine window size, using default value: -1")
        return -1

