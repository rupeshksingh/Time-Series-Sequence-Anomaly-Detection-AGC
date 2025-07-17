import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import time
from collections import deque
import io
import json
import pickle

# --- Import Core Logic ---
# TCNN (Deep Learning) specific imports
from core.model import train_anomaly_detector as train_tcnn_model
from core.processing import RealTimePredictor as RealTimePredictorTCNN

# XGBoost (Gradient Boosting) specific imports
from core.model_xgb import train_xgb_model
from core.processing_xgb import RealTimePredictorXGB

# Utility functions used by both
from core.utils import fill_anomaly_gaps

# --- Page and State Configuration ---
st.set_page_config(
    layout="wide",
    page_title="AGC Anomaly Detection",
    page_icon="assets/agc_icon.png"
)

# --- App State Initialization ---
if 'streaming' not in st.session_state:
    st.session_state.streaming = False
if 'data_stream' not in st.session_state:
    st.session_state.data_stream = deque(maxlen=500)
if 'raw_predictions' not in st.session_state:
    st.session_state.raw_predictions = []

# --- Asset Loading Functions ---
@st.cache_resource
def load_artifacts_tcnn(model_path, scaler_path, params_path):
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        with open(params_path, 'r') as f: params = json.load(f)
        if 'group_stats' in params:
            gs_dict = params['group_stats']
            params['group_stats'] = pd.DataFrame(gs_dict['data'], index=gs_dict['index'], columns=gs_dict['columns'])
        return {'model': model, 'scaler': scaler, 'params': params}
    except Exception as e:
        st.error(f"Error loading TCNN artifacts: {e}. Please ensure default files exist in `assets/`.")
        return None

@st.cache_resource
def load_artifacts_xgb(model_path, scaler_path, config_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(config_path, 'r') as f: config = json.load(f)
        return {'model': model, 'scaler': scaler, 'config': config}
    except Exception as e:
        st.error(f"Error loading XGBoost artifacts: {e}. Please ensure default files exist in `assets/`.")
        return None

def reset_streaming_state():
    st.session_state.streaming = False
    st.session_state.data_stream.clear()
    st.session_state.raw_predictions = []

# --- Main App UI ---
st.title("Time Series Anomaly Detection in the Glass Rolling Process")
st.markdown("A proof-of-concept dashboard developed as part of a data science project with **AGC Asia Pacific**.")

with st.expander("📘 Project Overview", expanded=False):
    st.markdown("""
        This application demonstrates a machine learning pipeline for monitoring a critical stage in glass manufacturing: the **glass rolling process**. 
        
        **Problem:** Anomalous fluctuations in roller speed often indicate glass breakage events, which can lead to significant production downtime (costing up to **$50,000 per hour**) and quality defects. Manual monitoring is labor-intensive and prone to error.
        
        **Solution:** This tool leverages advanced machine learning models to analyze high-frequency sensor data from the rollers in real-time. It aims to:
        - **Continuously monitor** roller speeds.
        - **Automatically detect** speed fluctuations caused by glass breakage.
        - **Identify the start and end** of each anomaly sequence for root cause analysis.
        
        By providing instant alerts and insights, this system enables operators to respond immediately, minimizing downtime and ensuring product quality.
    """)

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Controls & Configuration")
st.sidebar.image("assets/agc_icon.png", use_container_width=True)
model_type = st.sidebar.radio(
    "Select Model Architecture",
    ["TCNN (Deep Learning)", "XGBoost (Gradient Boosting)"],
    help="Choose the underlying model for detection. Each has its own strengths."
)
st.sidebar.divider()

# --- Display Model-Specific Information ---
if model_type == "TCNN (Deep Learning)":
    st.sidebar.info("""
        **TCNN (Temporal Convolutional Network):** A deep learning model that excels at capturing long-range temporal patterns and dependencies in sequential data, making it ideal for time series analysis.
    """)
    app_mode = st.sidebar.selectbox("Choose App Mode", ["Real-Time Prediction", "Train New Model"], key='tcnn_mode')
else:
    st.sidebar.info("""
        **XGBoost (Gradient Boosting):** A powerful and efficient tree-based model. It uses a rich set of statistical and engineered features to make highly accurate predictions, often outperforming other models on tabular data.
    """)
    app_mode = st.sidebar.selectbox("Choose App Mode", ["Real-Time Prediction", "Train New Model"], key='xgb_mode')

# =================================================================================================
# === TRAINING UI =================================================================================
# =================================================================================================
if app_mode == "Train New Model":
    if model_type == "TCNN (Deep Learning)":
        st.header("Train a New TCNN Model")
        train_file = st.file_uploader("Upload Labeled Training Data (CSV with 'speed' and 'pred_label')", type=['csv'], key='tcnn_train_upload')
        st.subheader("Hyperparameters")
        col1, col2, col3 = st.columns(3)
        seq_length = col1.number_input("Sequence Length", 30, 200, 80, key='tcnn_seq', help="Number of time steps in each sequence fed to the model.")
        epochs = col2.number_input("Epochs", 1, 100, 10, key='tcnn_epochs', help="Number of times the model will see the entire training dataset.")
        batch_size = col3.number_input("Batch Size", 32, 2048, 512, key='tcnn_batch', help="Number of sequences processed in each training step.")
        if st.button("Start TCNN Training", disabled=not train_file):
            df_train = pd.read_csv(train_file, usecols=["speed", "pred_label"])
            with st.spinner("Training TCNN model... This may take a while."):
                results = train_tcnn_model(df_train, seq_length=seq_length, epochs=epochs, batch_size=batch_size)
            st.success("✅ TCNN Model Training Completed!")
            
            tab1, tab2, tab3 = st.tabs(["📈 Performance Metrics", "📋 Classification Report", "💾 Download Artifacts"])
            with tab1:
                st.metric("Validation AUC", f"{results['val_auc']:.4f}")
                st.metric("Validation PR-AUC", f"{results['val_avg_precision']:.4f}")
            with tab2:
                st.json(results['report'])
            with tab3:
                st.markdown("Download the trained model and associated files.")
                
                # 1. Parameters
                params_to_save = {'seq_length': results['seq_length'], 'bins': results['bins'], 'group_stats': results['group_stats'].to_dict(orient='split'), 'feature_cols': results['feature_cols']}
                st.download_button(
                    label="1. Download Parameters (.json)", 
                    data=json.dumps(params_to_save, indent=2), 
                    file_name="tcnn_params.json",
                    mime="application/json"
                )

                # 2. Model
                # Save model to a temporary file path to satisfy Keras's requirement for a file extension
                model_path = "temp_tcnn_model.keras"
                results['model'].save(model_path)
                with open(model_path, "rb") as fp:
                    st.download_button(
                        label="2. Download TCNN Model (.keras)",
                        data=fp,
                        file_name="tcnn_model.keras",
                        mime="application/octet-stream"
                    )

                # 3. Scaler
                scaler_buffer = io.BytesIO()
                joblib.dump(results['scaler'], scaler_buffer)
                st.download_button(
                    label="3. Download Scaler (.pkl)",
                    data=scaler_buffer.getvalue(), # Use .getvalue() to pass the bytes
                    file_name="tcnn_scaler.pkl",
                    mime="application/octet-stream"
                )

    else: # XGBoost Training
        st.header("Train a New XGBoost Model")
        train_file = st.file_uploader("Upload Labeled Training Data (CSV with 'speed', 'pred_label', and 'indo_time')", type=['csv'], key='xgb_train_upload')
        if st.button("Start XGBoost Training", disabled=not train_file):
            df_train = pd.read_csv(train_file, parse_dates=['indo_time'])
            with st.spinner("Training XGBoost model... This may take a few minutes."):
                results = train_xgb_model(df_train)
            st.success("✅ XGBoost Model Training Completed!")

            tab1, tab2, tab3 = st.tabs(["📈 Performance & Visuals", "📋 Classification Report", "💾 Download Artifacts"])
            with tab1:
                st.pyplot(results['figure'])
            with tab2:
                st.json(results['report'])
            with tab3:
                st.download_button("1. Download Config (.json)", data=json.dumps(results['config'], indent=2), file_name="xgb_config.json")
                st.download_button("2. Download Model (.pkl)", data=pickle.dumps(results['model']), file_name="xgb_model.pkl")
                st.download_button("3. Download Scaler (.pkl)", data=pickle.dumps(results['scaler']), file_name="xgb_scaler.pkl")

# =================================================================================================
# === REAL-TIME PREDICTION UI =====================================================================
# =================================================================================================
elif app_mode == "Real-Time Prediction":
    st.header(f"{model_type.split(' ')[0]}: Live Process Monitoring")

    # --- Load artifacts based on selected model ---
    if model_type == "TCNN (Deep Learning)":
        artifacts = load_artifacts_tcnn('assets/default_model.keras', 'assets/default_scaler.pkl', 'assets/default_params.json')
    else:
        artifacts = load_artifacts_xgb('assets/xgb_model.pkl', 'assets/xgb_scaler.pkl', 'assets/xgb_config.json')

    unlabeled_file = st.sidebar.file_uploader("Upload Unlabeled Data (.csv)", type=['csv'], key=f'{model_type}_pred_upload')
    pred_threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.05, key=f'{model_type}_thresh', help="The probability score above which a point is classified as an anomaly. Lower this to detect more, potentially less severe, anomalies.")
    gap_fill_threshold = st.sidebar.slider("Gap Fill Threshold", 0, 300, 120, 10, key=f'{model_type}_gap', help="Connects anomaly sequences that are separated by a gap smaller than this number of points.")

    start_button = st.sidebar.button("Start Streaming", disabled=not (unlabeled_file and artifacts))
    stop_button = st.sidebar.button("Stop", disabled=not st.session_state.streaming)

    if st.session_state.streaming:
        st.sidebar.warning("🔴 Streaming in progress...")
        st.sidebar.info("To apply new settings, please Stop and Start the stream again.")

    if start_button:
        reset_streaming_state()
        st.session_state.streaming = True
        st.rerun()
    if stop_button:
        reset_streaming_state()
        st.rerun()

    # --- Main content area for prediction ---
    col1, col2 = st.columns([3, 1])
    with col1:
        chart_placeholder = st.empty()
    with col2:
        st.subheader("Live Stats")
        stats_placeholder = st.empty()

    if st.session_state.streaming:
        df = pd.read_csv(unlabeled_file)
        speed_col = 'A2:MCPGSpeed' if 'A2:MCPGSpeed' in df.columns else 'speed'
        
        if model_type == "TCNN (Deep Learning)":
            predictor = RealTimePredictorTCNN(model=artifacts['model'], scaler=artifacts['scaler'], **artifacts['params'])
        else: # XGBoost
            predictor = RealTimePredictorXGB(artifacts['model'], artifacts['scaler'], artifacts['config'])
        
        PLOT_UPDATE_INTERVAL = 5
        for index, row in df.iterrows():
            if not st.session_state.streaming:
                st.warning("Streaming stopped by user.")
                break
            
            speed_val = row[speed_col]
            st.session_state.data_stream.append(speed_val)
            prediction_proba, _ = predictor.process_new_point(speed_val)
            is_anomaly = (prediction_proba is not None and prediction_proba > pred_threshold)
            st.session_state.raw_predictions.append(1 if is_anomaly else 0)

            total_points = len(st.session_state.raw_predictions)
            total_anomalies = sum(st.session_state.raw_predictions)
            stats_placeholder.metric(label="Anomalies Detected", value=f"{total_anomalies}", delta=f"out of {total_points} points")

            if index % PLOT_UPDATE_INTERVAL == 0:
                fig_new = go.Figure()
                x_vals = list(range(len(st.session_state.data_stream)))
                fig_new.add_trace(go.Scatter(x=x_vals, y=list(st.session_state.data_stream), mode='lines', name='Speed', line=dict(color='royalblue')))
                anomaly_indices = [i for i, pred in enumerate(st.session_state.raw_predictions[-len(x_vals):]) if pred == 1]
                if anomaly_indices:
                    anomaly_values = [list(st.session_state.data_stream)[i] for i in anomaly_indices]
                    fig_new.add_trace(go.Scatter(x=anomaly_indices, y=anomaly_values, mode='markers', name='Anomaly Point', marker=dict(color='crimson', size=8)))
                
                filled = fill_anomaly_gaps(np.array(st.session_state.raw_predictions), gap_threshold=gap_fill_threshold)
                in_anomaly_seq, start_idx = False, 0
                for i in range(len(filled)):
                    if filled[i] == 1 and not in_anomaly_seq:
                        start_idx = i
                        in_anomaly_seq = True
                    elif filled[i] == 0 and in_anomaly_seq:
                        fig_new.add_vrect(x0=start_idx, x1=i-1, fillcolor="red", opacity=0.2, line_width=0, layer="below", name='Anomaly Sequence')
                        in_anomaly_seq = False
                if in_anomaly_seq:
                    fig_new.add_vrect(x0=start_idx, x1=len(filled)-1, fillcolor="red", opacity=0.2, line_width=0, layer="below", name='Anomaly Sequence')
                
                names = set()
                fig_new.for_each_trace(lambda trace: names.add(trace.name) if (trace.name in names) else trace.update(showlegend=True))
                fig_new.update_layout(title_text='Live Roller Speed Monitoring', xaxis_title='Time Step', yaxis_title='Roller Speed (RPM)')
                chart_placeholder.plotly_chart(fig_new, use_container_width=True)
            
            time.sleep(0.05)
        
        reset_streaming_state()
        st.success("✅ Streaming finished.")
    else:
        # Initial drawing of chart and stats when not streaming
        fig = go.Figure(go.Scatter(x=[], y=[], mode='lines', name='Speed'))
        fig.update_layout(title_text='Live Roller Speed Monitoring', xaxis_title='Time Step', yaxis_title='Roller Speed (RPM)')
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        stats_placeholder.metric(label="Anomalies Detected", value=0, delta="out of 0 points")
