import pandas as pd
import numpy as np
from collections import deque
from .realtime_feature_engineering import RealTimeFeatureEngineer

class RealTimePredictor:
    """
    Orchestrates the real-time prediction pipeline point-by-point.
    """
    def __init__(self, model, scaler, feature_cols, bins, group_stats, seq_length):
        self.model = model
        self.scaler = scaler
        self.bins = bins
        self.group_stats = group_stats
        self.seq_length = seq_length

        self.feature_engineer = RealTimeFeatureEngineer(feature_cols=feature_cols)

        self.sequence_buffer = deque(maxlen=self.seq_length)

        self.speed_buffer = deque(maxlen=5)

    def process_new_point(self, speed_value):
        """
        Processes a single data point through the entire pipeline.
        """
        self.speed_buffer.append(speed_value)
        speed_series = pd.Series(list(self.speed_buffer))

        current_bin = pd.cut(speed_series, bins=self.bins, labels=False, right=False).iloc[-1]
        current_bin = int(current_bin) if pd.notna(current_bin) else 0

        if current_bin in self.group_stats.index:
            mean = self.group_stats.loc[current_bin, 'mean']
            std = self.group_stats.loc[current_bin, 'std']
        else:
            mean, std = np.mean(self.speed_buffer), np.std(self.speed_buffer)

        speed_normalized = (speed_value - mean) / (std + 1e-6) if std > 0 else 0.0

        latest_features_dict = self.feature_engineer.update(speed_normalized)
        self.sequence_buffer.append(list(latest_features_dict.values()))

        if len(self.sequence_buffer) < self.seq_length:
            return None, None 

        feature_sequence = np.array(list(self.sequence_buffer))

        scaled_sequence = self.scaler.transform(feature_sequence)

        model_input = scaled_sequence.reshape(1, self.seq_length, -1)

        prediction_proba = self.model.predict(model_input, verbose=0)[0][0]
        
        return prediction_proba, (prediction_proba > 0.5)