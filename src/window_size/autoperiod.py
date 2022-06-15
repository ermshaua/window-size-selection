# coding=utf-8
from __future__ import division, print_function, absolute_import

import math

import numpy as np
import scipy
from astropy.timeseries import LombScargle
from scipy.signal import fftconvolve
from scipy.stats import linregress
from scipy import integrate
from six.moves import range


class Autoperiod(object):
    def __init__(self, times, values, plotter=None, threshold_method='mc', mc_iterations=40, confidence_level=.9):

        # convert absolute times to time differences from start timestamp
        self.times = times - times[0] if times[0] != 0 else times
        self.values = values
        self.plotter = plotter
        self.acf = self.autocorrelation(values)
        # normalize acf
        self.acf /= np.max(self.acf)
        self.acf *= 100

        self.time_span = self.times[-1] - self.times[0]
        self.time_interval = self.times[1] - self.times[0]

        freqs, self.powers = LombScargle(self.times, self.values).autopower(
            minimum_frequency=1 / self.time_span,
            maximum_frequency=1 / (self.time_interval * 2),
            normalization='psd'
        )
        self.periods = 1 / freqs

        self._power_norm_factor = 1 / (2 * np.var(self.values - np.mean(self.values)))
        # double the power, since the astropy lomb-scargle implementation halves it during the psd normalization
        self.powers = 2 * self.powers * self._power_norm_factor

        self._threshold_method = threshold_method
        self._mc_iterations = mc_iterations
        self._confidence_level = confidence_level
        self._power_threshold = self._get_power_threshold()

        self._period_hints = self._get_period_hints()

        period = None
        is_valid = False
        for i, p in self._period_hints:
            is_valid, period = self.validate_hint(i)
            if is_valid:
                break

        self._period = None
        self._sinwave = None

        if period and is_valid:
            self._period = period

            phase_shift = self.times[np.argmax(self.values)]
            amplitude = np.max(values) / 2
            self._sinwave = np.cos(2 * np.pi / self._period * (self.times - phase_shift)) * amplitude + amplitude

        if self.plotter:
            self.plotter.plot_timeseries(self.times, self.values)
            self.plotter.plot_acf(self.times, self.acf)
            if self._period:
                self.plotter.plot_sinwave(self.times, self._sinwave)
                self.plotter.plot_area_ratio(*self.period_area())

    @property
    def period(self):
        return self._period

    def period_blocks(self):
        period_region = self._sinwave > (np.max(self._sinwave) / 2)

        # An array of indices of the cutoff points for the period blocks, i.e. where it goes from
        # "on-period" to "off-period"
        period_cutoff_indices = np.where(period_region[:-1] != period_region[1:])[0] + 1

        # array([[times],
        #       [values]])
        timeseries = np.stack((self.times, self.values))

        period_blocks = np.array_split(timeseries, period_cutoff_indices, axis=1)

        on_period_blocks = period_blocks[::2] if period_region[0] else period_blocks[1::2]
        off_period_blocks = period_blocks[1::2] if period_region[0] else period_blocks[::2]

        return on_period_blocks, off_period_blocks

    def period_block_areas(self):
        on_period_blocks, off_period_blocks = self.period_blocks()
        on_block_areas = np.array([scipy.integrate.trapz(block[1], x=block[0]) for block in on_period_blocks])
        off_block_areas = np.array([scipy.integrate.trapz(block[1], x=block[0]) for block in off_period_blocks])
        return on_block_areas, off_block_areas

    def period_area(self):
        period_region = self._sinwave > (np.max(self._sinwave) / 2)

        on_period_area = integrate.trapz(self.values[period_region], self.times[period_region])
        off_period_area = integrate.trapz(self.values[~period_region], self.times[~period_region])

        return on_period_area, off_period_area

    @property
    def phase_shift_guess(self):
        return self.times[np.argmax(self.values)]

    @property
    def threshold_method(self):
        return self._threshold_method

    @threshold_method.setter
    def threshold_method(self, method):
        self._threshold_method = method
        self._power_threshold = self._get_power_threshold()
        self._threshold_method = method

    def _get_period_hints(self):
        period_hints = []

        total_periods = len(self.periods)
        for i, period in enumerate(self.periods):
            if self.powers[i] > self._power_threshold and self.time_span / 2 > period > 2 * self.time_interval:
                if i < (total_periods - 2) and i > 2:
                    period_hints.append((i, period))

        period_hints = sorted(period_hints, key=lambda p: self.powers[p[0]], reverse=True)

        if self.plotter:
            self.plotter.plot_periodogram(self.periods, self.powers, period_hints, self._power_threshold,
                                          self.time_span / 2)

        return period_hints

    def _get_power_threshold(self):
        if self.threshold_method == 'mc':
            return self._mc_threshold()
        elif self.threshold_method == 'stat':
            return self._stat_threshold()
        else:
            raise ValueError("Method must be one of 'mc', 'stat'")

    def _mc_threshold(self):
        max_powers = []
        shuf = np.copy(self.values)
        for _ in range(self._mc_iterations):
            np.random.shuffle(shuf)
            _, powers = LombScargle(self.times, shuf).autopower(normalization='psd')
            max_powers.append(np.max(powers))

        max_powers.sort()
        return max_powers[int(len(max_powers) * .99)] * self._power_norm_factor

    def _stat_threshold(self):
        return -1 * math.log(1 - math.pow(self._confidence_level, 1 / self.powers.size))

    def validate_hint(self, period_idx):
        search_min, search_max = self._get_acf_search_range(period_idx)

        min_err = float("inf")
        t_split = None
        min_slope1 = 0
        min_slope2 = 0
        for t in range(search_min + 1, search_max):
            seg1_x = self.times[search_min:t + 1]
            seg1_y = self.acf[search_min:t + 1]
            seg2_x = self.times[t:search_max + 1]
            seg2_y = self.acf[t:search_max + 1]

            slope1, c1, _, _, stderr1 = linregress(seg1_x, seg1_y)
            slope2, c2, _, _, stderr2 = linregress(seg2_x, seg2_y)

            if stderr1 + stderr2 < min_err and seg1_x.size > 2 and seg2_x.size > 2:
                min_err = stderr1 + stderr2
                t_split = t
                min_slope1 = slope1
                min_slope2 = slope2
                min_c1 = c1
                min_c2 = c2
                min_stderr1 = stderr1
                min_stderr2 = stderr2

        angle1 = np.arctan(min_slope1) / (np.pi / 2)
        angle2 = np.arctan(min_slope2) / (np.pi / 2)
        valid = min_slope1 > min_slope2 and not np.isclose(np.abs(angle1 - angle2), 0, atol=0.01)
        window = self.acf[search_min:search_max + 1]
        peak_idx = np.argmax(window) + search_min

        if self.plotter and (valid or self.plotter.verbose):
            self.plotter.plot_acf_validation(
                self.times,
                self.acf,
                self.times[search_min:t_split + 1], min_slope1, min_c1, min_stderr1,
                self.times[t_split:search_max + 1], min_slope2, min_c2, min_stderr2,
                t_split, peak_idx
            )

        return valid, self.times[peak_idx]

    def _get_acf_search_range(self, period_idx):
        min_period = 0.5 * (self.periods[period_idx + 1] + self.periods[period_idx + 2])
        max_period = 0.5 * (self.periods[period_idx - 1] + self.periods[period_idx - 2])

        min_idx = self.closest_index(min_period, self.times)
        max_idx = self.closest_index(max_period, self.times)
        while max_idx - min_idx + 1 < 6:
            if min_idx > 0:
                min_idx -= 1
            if max_idx < self.times.size - 1:
                max_idx += 1

        return min_idx, max_idx

    @staticmethod
    def autocorrelation(values):
        acf = fftconvolve(values, values[::-1], mode='full')
        return acf[acf.size // 2:]

    @staticmethod
    def closest_index(value, arr):
        return np.abs(arr - value).argmin()


def autoperiod(ts):
    try:
        window_size = Autoperiod(np.arange(ts.shape[0]), ts).period
        return np.int64(window_size)
    except:
        print(f"Could not determine window size, using default value: -1")
        return -1