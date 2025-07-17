import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
import gc
import streamlit as st

from .feature_engineering import create_time_series_features_advanced

def create_sequences(X, y, seq_length):
    sequences, labels = [], []
    for i in range(len(X) - seq_length + 1):
        sequences.append(X[i:i + seq_length])
        labels.append(y[i + seq_length - 1])
    return np.array(sequences), np.array(labels)

def build_simple_tcnn_model(seq_length, n_features):
    inputs = layers.Input(shape=(seq_length, n_features))
    x = layers.Conv1D(64, 5, padding='causal', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, 7, padding='causal', dilation_rate=2, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, 9, padding='causal', dilation_rate=4, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

@st.cache_data
def train_anomaly_detector(_df, seq_length=120, epochs=10, batch_size=512, learning_rate=0.001):
    tf.keras.backend.clear_session()
    gc.collect()

    df = _df.copy()
    bins = [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000]
    df["speed_bin"] = pd.cut(df["speed"], bins=bins, labels=False)
    df["speed_bin"] = df["speed_bin"].astype(int)
    group_stats = df.groupby("speed_bin")["speed"].agg(["mean", "std"])
    df = df.merge(group_stats, how="left", left_on="speed_bin", right_index=True)
    df["speed_normalized"] = (df["speed"] - df["mean"]) / df["std"]

    df_features = create_time_series_features_advanced(df, target_col='speed_normalized')
    feature_cols = [col for col in df_features.columns if col not in ['speed', 'pred_label', 'time_value', 'mean', 'std', 'speed_bin', 'speed_normalized']]
    
    X = df_features[feature_cols].values
    y = df_features['pred_label'].values
    X = pd.DataFrame(X, columns=feature_cols).ffill().bfill().values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq, y_seq = create_sequences(X_scaled, y, seq_length)
    
    split_idx = int(len(X_seq) * 0.95)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    model = build_simple_tcnn_model(seq_length=seq_length, n_features=X_train.shape[2])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=0.2),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    early_stop = callbacks.EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max')
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    y_val_pred_proba = model.predict(X_val, batch_size=batch_size, verbose=0).flatten()
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_avg_precision = average_precision_score(y_val, y_val_pred_proba)
    report = classification_report(y_val, (y_val_pred_proba > 0.5).astype(int), output_dict=True)

    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'bins': bins,
        'group_stats': group_stats,
        'seq_length': seq_length,
        'history': history.history,
        'val_auc': val_auc,
        'val_avg_precision': val_avg_precision,
        'report': report
    }