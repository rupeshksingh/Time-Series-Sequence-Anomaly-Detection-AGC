import pandas as pd
from .realtime_feature_engineering_xgb import RealTimeFeatureEngineerXGB

class RealTimePredictorXGB:
    def __init__(self, model, scaler, config):
        self.model = model
        self.scaler = scaler
        self.config = config
        self.feature_names = config['feature_names']
        self.feature_engineer = RealTimeFeatureEngineerXGB(self.feature_names)

    def process_new_point(self, speed_value):
        # 1. Real-time Feature Engineering
        features_dict = self.feature_engineer.update(speed_value)
        
        # 2. Convert to DataFrame to ensure correct feature names for scaler
        feature_df = pd.DataFrame([features_dict], columns=self.feature_names)
        
        # 3. Scale features
        scaled_features = self.scaler.transform(feature_df)
        
        # 4. Predict
        prediction_proba = self.model.predict_proba(scaled_features)[0, 1]
        
        return prediction_proba, (prediction_proba > 0.5)
