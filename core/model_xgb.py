# core/model_xgb.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_curve, auc, classification_report
from sklearn.preprocessing import RobustScaler
from datetime import datetime
import joblib
import json
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from .feature_engineering_xgb import engineer_robust_features

@st.cache_data
def train_xgb_model(_df, use_temporal_validation=True, random_state=42):
    """
    Trains, validates, and saves an XGBoost anomaly detection model.
    """
    df = _df.copy()
    speed_col='speed'
    label_col='pred_label'
    timestamp_col='indo_time' if 'indo_time' in df.columns else None

    st.write("Step 1: Engineering robust features...")
    X = engineer_robust_features(df, speed_col)
    y = df[label_col]
    
    # --- Data Splitting ---
    if use_temporal_validation and timestamp_col:
        st.write("Step 2: Performing temporal train-validation split (80/20)...")
        sorted_idx = df.sort_values(timestamp_col).index
        X_sorted, y_sorted = X.loc[sorted_idx], y.loc[sorted_idx]
        train_end = int(0.8 * len(X_sorted))
        X_train, y_train = X_sorted.iloc[:train_end], y_sorted.iloc[:train_end]
        X_val, y_val = X_sorted.iloc[train_end:], y_sorted.iloc[train_end:]
    else:
        st.write("Step 2: Performing stratified train-validation split (80/20)...")
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

    # --- Scaling ---
    st.write("Step 3: Scaling features with RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # --- Model Training ---
    st.write("Step 4: Training XGBoost model with RandomizedSearchCV...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
    
    param_grid = {
        'n_estimators': [100, 150], 'max_depth': [4, 5, 6], 'learning_rate': [0.05, 0.1],
        'min_child_weight': [5, 10], 'gamma': [0.1, 0.5], 'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8], 'scale_pos_weight': [scale_pos_weight]
    }
    
    base_model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss', early_stopping_rounds=10, n_jobs=-1)
    cv_strategy = TimeSeriesSplit(n_splits=3) if use_temporal_validation else 3
    
    random_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=5, scoring='f1', cv=cv_strategy, verbose=1, random_state=random_state, n_jobs=-1)
    random_search.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    
    model = random_search.best_estimator_

    # --- Evaluation ---
    st.write("Step 5: Evaluating model on validation set...")
    y_val_pred = model.predict(X_val_scaled)
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    report = classification_report(y_val, y_val_pred, output_dict=True)
    
    # --- Artifacts ---
    config = {
        'feature_names': list(X.columns),
        'model_params': model.get_params(),
        'training_date': datetime.now().isoformat(),
        'temporal_validation': use_temporal_validation,
    }

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm = confusion_matrix(y_val, y_val_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    
    if len(np.unique(y_val)) > 1:
        fpr, tpr, _ = roc_curve(y_val, y_val_proba)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, 'b-', label=f'AUC = {roc_auc:.3f}')
        axes[1].plot([0, 1], [0, 1], 'r--')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
    plt.tight_layout()

    return {
        'model': model,
        'scaler': scaler,
        'config': config,
        'report': report,
        'figure': fig
    }