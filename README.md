# Real-Time Time Series Anomaly Detection UI

![Application UI](assets\UI.png)

This application provides a comprehensive user interface for training and performing real-time anomaly detection on time series data using one of two powerful machine learning architectures.

## 🚀 Key Features

-   **Dual Model Architecture**: Choose between two distinct anomaly detection engines:
    -   **TCNN (Temporal Convolutional Neural Network)**: A deep learning model ideal for capturing complex sequential patterns and temporal dependencies.
    -   **XGBoost (Gradient Boosting)**: A robust tree-based model that uses a rich set of statistical and engineered features for high performance.

-   **Real-Time Prediction**: Stream unlabeled data and visualize anomalies as they are detected, with live updates to charts and statistics.

-   **Customizable Engine**: Use the default pre-trained models or upload your own. Adjust prediction thresholds and post-processing parameters on the fly.

-   **Interactive Visualization**: A live Plotly chart shows the speed data, detected anomaly points, and consolidated anomaly sequences for intuitive analysis.

-   **Model Training from UI**: Train new TCNN or XGBoost models directly within the application using your own labeled datasets and custom hyperparameters.

-   **Download Artifacts**: Save your newly trained models, data scalers, and configuration files for future use or deployment.

## ⚙️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd anomaly-detection-ui
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Place Data and Asset Files:**
    -   Put your sample training data (e.g., `event_all.csv`) in the `data/` folder.
    -   Put your sample unlabeled data for prediction (e.g., `evaluate.csv`) in the `data/` folder.
    -   Ensure the default pre-trained model assets are in the `assets/` directory:
        -   **For TCNN**: `default_model.keras`, `default_scaler.pkl`, `default_params.json`
        -   **For XGBoost**: `xgb_model.pkl`, `xgb_scaler.pkl`, `xgb_config.json`

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
5.  Open your browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`). Use the sidebar to select your desired model architecture and mode (Prediction or Training).

