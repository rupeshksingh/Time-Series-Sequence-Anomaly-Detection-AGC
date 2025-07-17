import numpy as np
import pandas as pd
from collections import deque
from scipy import signal

class RealTimeFeatureEngineerXGB:
    """
    A stateful feature engineer for the XGBoost model that correctly calculates
    all features on a point-by-point basis for real-time prediction. This is a
    complete implementation matching the batch script.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.max_window = 1200
        
        # A single, large buffer for the raw speed points
        self.speed_buffer = deque(maxlen=self.max_window)
        
        # States for various feature calculations
        self.ewm_states = {
            'alpha_0.1': {'mean': 0.0, 'is_first': True},
            'alpha_0.5': {'mean': 0.0, 'is_first': True},
            'alpha_0.9': {'mean': 0.0, 'is_first': True},
            'span_60': {'mean': 0.0, 'var': 0.0, 'is_first': True},
            'span_450': {'mean': 0.0, 'var': 0.0, 'is_first': True},
            'span_900': {'mean': 0.0, 'var': 0.0, 'is_first': True},
        }
        self.lag_buffers = {lag: deque(maxlen=lag + 1) for lag in [1, 5, 10, 30, 60]}
        self.cumsum_change = 0.0
        self.kalman_x_est, self.kalman_p_est = 0.0, 1.0
        self.is_first_kalman = True
        self.expanding_q05 = 0.0
        self.expanding_q95 = 0.0
        self.point_count = 0
        self.prev_derivative_1 = 0.0

    def update(self, new_point):
        self.speed_buffer.append(new_point)
        self.point_count += 1
        features = {}
        
        # --- EWM Alpha Features ---
        for alpha in [0.1, 0.5, 0.9]:
            state = self.ewm_states[f'alpha_{alpha}']
            if state['is_first']:
                state['mean'] = new_point
                state['is_first'] = False
            else:
                state['mean'] = alpha * new_point + (1 - alpha) * state['mean']
            features[f'Speed_ema_{alpha}'] = state['mean']

        # --- Savitzky-Golay ---
        for w in [30, 180, 600]:
            if len(self.speed_buffer) >= w:
                arr = np.array(list(self.speed_buffer)[-w:])
                # Use polyorder that is safe for the window size
                polyorder = min(5, w - 2) if w > 2 else 1
                features[f'Speed_savgol_{w}'] = signal.savgol_filter(arr, w, polyorder)[-1]
            else:
                features[f'Speed_savgol_{w}'] = 0.0

        # --- Rolling Window Features ---
        for w in [30, 60, 120, 300, 600, 900, 1200]:
            # Use full buffer if available, otherwise partial
            arr = np.array(self.speed_buffer) if len(self.speed_buffer) < w else np.array(list(self.speed_buffer)[-w:])

            if w in [60, 300, 600, 900, 1200]:
                median = np.median(arr)
                mad = np.median(np.abs(arr - median))
                q25, q75 = np.quantile(arr, [0.25, 0.75])
                iqr = q75 - q25
                features[f'Speed_rolling_median_{w}'] = median
                features[f'Speed_rolling_mad_{w}'] = mad
                features[f'Speed_robust_zscore_{w}'] = (new_point - median) / (mad + 1e-9) if mad > 0 else 0
                features[f'Speed_rolling_q25_{w}'] = q25
                features[f'Speed_rolling_q75_{w}'] = q75
                features[f'Speed_rolling_iqr_{w}'] = iqr
                features[f'Speed_outside_iqr_{w}'] = ((new_point < (q25 - 1.5 * iqr)) or (new_point > (q75 + 1.5 * iqr))).astype(int)

            if w in [30, 60, 120]:
                if len(arr) > 1:
                    diffs = np.diff(arr)
                    features[f'Speed_increases_in_{w}'] = np.sum(diffs > 0)
                    features[f'Speed_decreases_in_{w}'] = np.sum(diffs < 0)
                else:
                    features[f'Speed_increases_in_{w}'] = 0
                    features[f'Speed_decreases_in_{w}'] = 0
                features[f'Speed_rolling_var_{w}'] = np.var(arr)

            if w in [60, 300]:
                rolling_min, rolling_max = np.min(arr), np.max(arr)
                features[f'Speed_is_window_max_{w}'] = (new_point == rolling_max).astype(int)
                features[f'Speed_is_window_min_{w}'] = (new_point == rolling_min).astype(int)
                range_val = rolling_max - rolling_min
                features[f'Speed_range_{w}'] = range_val
                features[f'Speed_range_position_{w}'] = ((new_point - rolling_min) / (range_val + 1e-9)) if range_val > 0 else 0

        # --- Lag Features ---
        for lag in [1, 5, 10, 30, 60]:
            self.lag_buffers[lag].append(new_point)
            if len(self.lag_buffers[lag]) > lag:
                prev_point = self.lag_buffers[lag][0]
                features[f'Speed_pct_change_{lag}'] = (new_point - prev_point) / (prev_point + 1e-9) if prev_point != 0 else 0
                features[f'Speed_abs_change_{lag}'] = np.abs(new_point - prev_point)
                features[f'Speed_rate_of_change_{lag}'] = (new_point - prev_point) / lag
            else:
                features[f'Speed_pct_change_{lag}'] = 0.0
                features[f'Speed_abs_change_{lag}'] = 0.0
                features[f'Speed_rate_of_change_{lag}'] = 0.0

        # --- Cumulative/Expanding Features ---
        if len(self.speed_buffer) > 1:
            self.cumsum_change += self.speed_buffer[-1] - self.speed_buffer[-2]
        features['Speed_cumsum_change'] = self.cumsum_change

        if self.point_count >= 100:
            self.expanding_q05, self.expanding_q95 = np.quantile(np.array(self.speed_buffer), [0.05, 0.95])
        features['Speed_below_historical_q05'] = int(new_point < self.expanding_q05)
        features['Speed_above_historical_q95'] = int(new_point > self.expanding_q95)

        # --- EWM Span Features ---
        for span in [60, 450, 900]:
            state = self.ewm_states[f'span_{span}']
            alpha = 2 / (span + 1)
            if state['is_first']:
                state['mean'] = new_point
                state['var'] = 0
                state['is_first'] = False
            else:
                old_mean = state['mean']
                state['mean'] = alpha * new_point + (1 - alpha) * state['mean']
                state['var'] = (1 - alpha) * (state['var'] + alpha * (new_point - old_mean)**2)
            
            ewm_std = np.sqrt(state['var'])
            features[f'Speed_ewm_mean_{span}'] = state['mean']
            features[f'Speed_ewm_std_{span}'] = ewm_std
            features[f'Speed_ewm_zscore_{span}'] = (new_point - state['mean']) / (ewm_std + 1e-9) if ewm_std > 0 else 0

        # --- Derivative Features ---
        speed_smooth = features['Speed_ema_0.5']
        # The first derivative is the difference from the previous smoothed value
        if len(self.speed_buffer) > 1:
            # Reconstruct previous EWM value before it was updated with new_point
            prev_ema_05 = (self.ewm_states['alpha_0.5']['mean'] - 0.5 * new_point) / 0.5
            deriv1 = speed_smooth - prev_ema_05
        else:
            deriv1 = 0
        features['Speed_derivative_1'] = deriv1
        
        # The second derivative is the difference from the previous first derivative
        deriv2 = deriv1 - self.prev_derivative_1
        features['Speed_derivative_2'] = deriv2
        self.prev_derivative_1 = deriv1 # Update state for the next point

        # --- Kalman Filter ---
        if self.is_first_kalman:
            self.kalman_x_est = new_point
            self.is_first_kalman = False
        
        x_pred = self.kalman_x_est
        p_pred = self.kalman_p_est + 1e-5
        k_gain = p_pred / (p_pred + 0.1)
        self.kalman_x_est = x_pred + k_gain * (new_point - x_pred)
        self.kalman_p_est = (1 - k_gain) * p_pred
        
        features['Speed_kalman_filtered'] = self.kalman_x_est
        features['Speed_kalman_residual'] = new_point - self.kalman_x_est
        features['Speed_kalman_residual_abs'] = np.abs(new_point - self.kalman_x_est)

        # Ensure all expected features are present and in the correct order
        final_features = {k: features.get(k, 0.0) for k in self.feature_names}
        return final_features