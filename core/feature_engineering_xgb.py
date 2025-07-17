# core/feature_engineering_xgb.py
import pandas as pd
import numpy as np
from scipy import signal

# The function engineer_robust_features from your provided script goes here.
# (No changes are needed to the function itself.)
def engineer_robust_features(df, speed_col='speed'):
    """
    Enhanced feature engineering without data leakage for irregular sensor data.
    All features only use past information (causal features).
    """
    features = pd.DataFrame(index=df.index)
    features['Speed'] = df[speed_col].copy()

    alphas = [0.1, 0.5, 0.9]
    for alpha in alphas:
        features[f'Speed_ema_{alpha}'] = df[speed_col].ewm(alpha=alpha, adjust=True).mean()
    
    for window_length in [30, 180, 600]:
        if len(df) > window_length:
            padded = np.pad(df[speed_col].values, (window_length-1, 0), mode='edge')
            filtered = signal.savgol_filter(padded, window_length, min(5, window_length-2))
            features[f'Speed_savgol_{window_length}'] = filtered[window_length-1:]

    windows = [60, 300, 600, 900, 1200]
    
    for window in windows:
        
        features[f'Speed_rolling_median_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=1).median()
        )
        
        def causal_mad(series, window):
            mad_values = []
            for i in range(len(series)):
                start_idx = max(0, i - window + 1)
                window_data = series.iloc[start_idx:i+1]
                if len(window_data) > 0:
                    median = window_data.median()
                    mad = (window_data - median).abs().median()
                    mad_values.append(mad)
                else:
                    mad_values.append(np.nan)
            return pd.Series(mad_values, index=series.index)
        
        features[f'Speed_rolling_mad_{window}'] = causal_mad(df[speed_col], window)
        
        median_col = features[f'Speed_rolling_median_{window}']
        mad_col = features[f'Speed_rolling_mad_{window}']
        features[f'Speed_robust_zscore_{window}'] = (
            (df[speed_col] - median_col) / (mad_col + 1e-9)
        )
        
        features[f'Speed_rolling_q25_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=1).quantile(0.25)
        )
        features[f'Speed_rolling_q75_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=1).quantile(0.75)
        )
        features[f'Speed_rolling_iqr_{window}'] = (
            features[f'Speed_rolling_q75_{window}'] - features[f'Speed_rolling_q25_{window}']
        )
        
        lower_bound = features[f'Speed_rolling_q25_{window}'] - 1.5 * features[f'Speed_rolling_iqr_{window}']
        upper_bound = features[f'Speed_rolling_q75_{window}'] + 1.5 * features[f'Speed_rolling_iqr_{window}']
        features[f'Speed_outside_iqr_{window}'] = (
            ((df[speed_col] < lower_bound) | (df[speed_col] > upper_bound)).astype(int)
        )
    
    for lag in [1, 5, 10, 30, 60]:
        features[f'Speed_pct_change_{lag}'] = df[speed_col].pct_change(periods=lag).fillna(0)

        features[f'Speed_abs_change_{lag}'] = df[speed_col].diff(periods=lag).abs().fillna(0)

        features[f'Speed_rate_of_change_{lag}'] = (
            df[speed_col].diff(periods=lag).fillna(0) / lag
        )

    features['Speed_cumsum_change'] = df[speed_col].diff().fillna(0).cumsum()

    for window in [30, 60, 120]:
        def count_increases(series, window):
            increases = []
            for i in range(len(series)):
                start_idx = max(0, i - window + 1)
                window_data = series.iloc[start_idx:i+1]
                if len(window_data) > 1:
                    diffs = window_data.diff()
                    increases.append((diffs > 0).sum())
                else:
                    increases.append(0)
            return pd.Series(increases, index=series.index)
        
        features[f'Speed_increases_in_{window}'] = count_increases(df[speed_col], window)
        features[f'Speed_decreases_in_{window}'] = window - features[f'Speed_increases_in_{window}']

        features[f'Speed_rolling_var_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=1).var().fillna(0)
        )

    historical_q05 = df[speed_col].expanding(min_periods=100).quantile(0.05)
    historical_q95 = df[speed_col].expanding(min_periods=100).quantile(0.95)
    
    features['Speed_below_historical_q05'] = (df[speed_col] < historical_q05).astype(int)
    features['Speed_above_historical_q95'] = (df[speed_col] > historical_q95).astype(int)

    for span in [60, 450, 900]:
        ewm = df[speed_col].ewm(span=span, adjust=True)
        features[f'Speed_ewm_mean_{span}'] = ewm.mean()
        features[f'Speed_ewm_std_{span}'] = ewm.std().fillna(0)

        features[f'Speed_ewm_zscore_{span}'] = (
            (df[speed_col] - features[f'Speed_ewm_mean_{span}']) / 
            (features[f'Speed_ewm_std_{span}'] + 1e-9)
        )

    for window in [60, 300]:
        features[f'Speed_is_window_max_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=1).max() == df[speed_col]
        ).astype(int)

        features[f'Speed_is_window_min_{window}'] = (
            df[speed_col].rolling(window=window, min_periods=1).min() == df[speed_col]
        ).astype(int)

    speed_smooth = features['Speed_ema_0.5']
    features['Speed_derivative_1'] = speed_smooth.diff().fillna(0)

    features['Speed_derivative_2'] = features['Speed_derivative_1'].diff().fillna(0)

    for window in [60, 300]:
        rolling_min = df[speed_col].rolling(window=window, min_periods=1).min()
        rolling_max = df[speed_col].rolling(window=window, min_periods=1).max()
        
        features[f'Speed_range_{window}'] = rolling_max - rolling_min

        features[f'Speed_range_position_{window}'] = (
            (df[speed_col] - rolling_min) / (features[f'Speed_range_{window}'] + 1e-9)
        ).clip(0, 1)

    def apply_kalman_filter(signal, process_variance=1e-5, measurement_variance=0.1):
        """Simple 1D Kalman filter"""
        n = len(signal)
        filtered = np.zeros(n)

        x_est = signal.iloc[0] if not pd.isna(signal.iloc[0]) else 0
        p_est = 1.0
        
        for i in range(n):
            x_pred = x_est
            p_pred = p_est + process_variance

            if not pd.isna(signal.iloc[i]):
                k_gain = p_pred / (p_pred + measurement_variance)
                x_est = x_pred + k_gain * (signal.iloc[i] - x_pred)
                p_est = (1 - k_gain) * p_pred
            
            filtered[i] = x_est
        
        return filtered
    
    kalman_filtered = apply_kalman_filter(df[speed_col])
    features['Speed_kalman_filtered'] = kalman_filtered
    features['Speed_kalman_residual'] = df[speed_col] - kalman_filtered
    features['Speed_kalman_residual_abs'] = np.abs(features['Speed_kalman_residual'])

    features = features.fillna(0)
    features = features.drop('Speed', axis=1)
    
    return features