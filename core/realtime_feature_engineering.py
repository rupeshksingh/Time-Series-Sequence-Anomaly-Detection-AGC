import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import iqr, entropy

class RealTimeFeatureEngineer:
    """
    Manages stateful, point-by-point feature engineering for real-time data.
    """
    def __init__(self, feature_cols, window_70=70, window_180=180, alpha=0.95):
        self.feature_cols = feature_cols
        self.w70 = window_70
        self.w180 = window_180
        self.alpha = alpha

        self.buffer70 = deque(maxlen=self.w70)
        self.buffer180 = deque(maxlen=self.w180)

        self.diff_buffer = deque(maxlen=2)
        self.accel_buffer = deque(maxlen=3)
        self.vol_of_vol_buffer = deque(maxlen=self.w70)

        self.ewm_mean = 0.0
        self.ewm_var = 0.0
        self.is_first_point = True

    def _get_val(self, arr, func, default=0.0):
        return func(arr) if len(arr) > 0 else default

    def _get_std(self, arr, default=0.0):
        return np.std(arr, ddof=1) if len(arr) > 1 else default

    def update(self, new_point):
        """
        Updates all features with one new data point and returns a feature dictionary.
        """
        self.buffer70.append(new_point)
        self.buffer180.append(new_point)
        self.diff_buffer.append(new_point)
        self.accel_buffer.append(new_point)

        features = {}
        arr70 = np.array(self.buffer70)
        arr180 = np.array(self.buffer180)

        mean70 = self._get_val(arr70, np.mean)
        std70 = self._get_std(arr70)
        features[f'speed_normalized_rolling_mean_{self.w70}'] = mean70
        features[f'speed_normalized_rolling_std_{self.w70}'] = std70
        features[f'speed_normalized_rolling_max_{self.w70}'] = self._get_val(arr70, np.max)
        features[f'speed_normalized_rolling_iqr_{self.w70}'] = self._get_val(arr70, iqr, default=0.0)

        if self.is_first_point:
            self.ewm_mean = new_point
            self.ewm_var = 0
            self.is_first_point = False
        else:
            old_mean = self.ewm_mean
            self.ewm_mean = self.alpha * new_point + (1 - self.alpha) * self.ewm_mean
            self.ewm_var = (1 - self.alpha) * (self.ewm_var + self.alpha * (new_point - old_mean)**2)

        ewm_std = np.sqrt(self.ewm_var)
        features[f'speed_normalized_ewm_mean_{self.alpha}'] = self.ewm_mean
        features[f'speed_normalized_ewm_std_{self.alpha}'] = ewm_std

        features[f'speed_normalized_diff_abs_1'] = np.abs(self.diff_buffer[1] - self.diff_buffer[0]) if len(self.diff_buffer) == 2 else 0.0
        if len(self.accel_buffer) == 3:
            accel = (self.accel_buffer[2] - self.accel_buffer[1]) - (self.accel_buffer[1] - self.accel_buffer[0])
        else:
            accel = 0.0
        features['speed_normalized_acceleration'] = accel

        if len(self.buffer180) == self.w180:
            trend = np.polyfit(np.arange(self.w180), arr180, 1)[0]
            fft_vals = np.fft.rfft(arr180)
            fft_mag = np.abs(fft_vals)
            fft_freq = np.fft.rfftfreq(self.w180)
            centroid = np.sum(fft_mag * fft_freq) / np.sum(fft_mag) if np.sum(fft_mag) > 0 else 0
            psd = fft_mag**2
            psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else np.zeros_like(psd)
            entropy_val = entropy(psd_norm)
        else:
            trend, centroid, entropy_val = 0.0, 0.0, 0.0

        features[f'speed_normalized_trend_{self.w180}'] = trend
        features[f'speed_normalized_fft_spectral_centroid_{self.w180}'] = centroid
        features[f'speed_normalized_fft_spectral_entropy_{self.w180}'] = entropy_val

        features[f'speed_normalized_zscore_{self.w70}'] = (new_point - mean70) / (std70 + 1e-6) if std70 > 0 else 0.0
        features[f'speed_normalized_to_rolling_mean_ratio'] = new_point / (mean70 + 1e-6) if mean70 != 0 else 0.0

        self.vol_of_vol_buffer.append(ewm_std)
        features[f'speed_normalized_vol_of_vol_{self.w70}'] = self._get_std(np.array(self.vol_of_vol_buffer))

        return pd.Series(features)[self.feature_cols].to_dict()