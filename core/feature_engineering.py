import numpy as np
from scipy.stats import entropy, iqr
import pywt
from scipy.signal import find_peaks

def cwt_energy(series):
    wavelet_name = 'mexh'
    wavelet_widths = [15]
    coefficients, frequencies = pywt.cwt(series.values, wavelet_widths, wavelet_name)
    return np.sum(coefficients[0]**2)

def count_peaks(series):
    peaks, _ = find_peaks(series.values)
    return len(peaks)

def rolling_fft_features(series, window):
    spectral_centroids = []
    spectral_entropies = []
    for i in range(len(series)):
        if i < window - 1:
            spectral_centroids.append(np.nan)
            spectral_entropies.append(np.nan)
        else:
            window_data = series.iloc[i-window+1:i+1].values
            if not np.isnan(window_data).any():
                fft_vals = np.fft.rfft(window_data)
                fft_mag = np.abs(fft_vals)
                fft_freq = np.fft.rfftfreq(len(window_data))
                centroid = np.sum(fft_mag * fft_freq) / np.sum(fft_mag) if np.sum(fft_mag) > 0 else 0
                spectral_centroids.append(centroid)
                psd = fft_mag**2
                psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else np.zeros_like(psd)
                spec_entropy = entropy(psd_norm) if len(psd_norm[psd_norm > 0]) > 0 else 0
                spectral_entropies.append(spec_entropy)
            else:
                spectral_centroids.append(np.nan)
                spectral_entropies.append(np.nan)
    return spectral_centroids, spectral_entropies

def create_time_series_features_advanced(df, target_col='speed_normalized', has_label=True):
    df = df.copy()
    window = 70
    df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
    df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
    df[f'{target_col}_rolling_iqr_{window}'] = df[target_col].rolling(window=window).apply(iqr, raw=True)
    alpha = 0.95
    df[f'{target_col}_ewm_mean_{alpha}'] = df[target_col].ewm(alpha=alpha).mean()
    df[f'{target_col}_ewm_std_{alpha}'] = df[target_col].ewm(alpha=alpha).std()
    diff_period = 1
    df[f'{target_col}_diff_abs_{diff_period}'] = df[target_col].diff(diff_period).abs()
    trend_window = 180
    slopes = [np.polyfit(np.arange(trend_window), df[target_col].iloc[i-trend_window+1:i+1], 1)[0]
              if i >= trend_window - 1 and not df[target_col].iloc[i-trend_window+1:i+1].isnull().any()
              else np.nan
              for i in range(len(df))]
    df[f'{target_col}_trend_{trend_window}'] = slopes
    df[f'{target_col}_acceleration'] = df[target_col].diff().diff()
    fft_window = 180
    spectral_centroids, spectral_entropies = rolling_fft_features(df[target_col], fft_window)
    df[f'{target_col}_fft_spectral_centroid_{fft_window}'] = spectral_centroids
    df[f'{target_col}_fft_spectral_entropy_{fft_window}'] = spectral_entropies
    df[f'{target_col}_zscore_{window}'] = (df[target_col] - df[f'{target_col}_rolling_mean_{window}']) / (df[f'{target_col}_rolling_std_{window}'] + 1e-6)
    df[f'{target_col}_to_rolling_mean_ratio'] = df[target_col] / (df[f'{target_col}_rolling_mean_{window}'] + 1e-6)
    vol_of_vol_window = 70
    if f'{target_col}_ewm_std_0.95' in df.columns:
        df[f'{target_col}_vol_of_vol_{vol_of_vol_window}'] = df[f'{target_col}_ewm_std_0.95'].rolling(window=vol_of_vol_window).std()
    return df