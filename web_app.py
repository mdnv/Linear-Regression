import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from train_model import (
    EXCLUDED_COLUMNS,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    PLOT_PATH,
    TARGET,
    build_training_frame,
    load_dataset,
    train_and_save_artifacts,
)

st.set_page_config(page_title="Wine Density Model", page_icon="W", layout="wide")

BASE_DIR = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_data(show_spinner=False)
def get_training_frame() -> pd.DataFrame:
    return build_training_frame(get_dataset())


@st.cache_resource(show_spinner=False)
def get_artifacts() -> tuple[Any, dict]:
    artifacts = train_and_save_artifacts()
    return artifacts["model"], artifacts["metrics"]


def format_label(name: str) -> str:
    return name.replace("_", " ").title()


def load_metrics_from_disk() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return get_artifacts()[1]


def load_model_from_disk():
    if MODEL_PATH.exists():
        with MODEL_PATH.open("rb") as file_obj:
            return pickle.load(file_obj)
    return get_artifacts()[0]


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(217, 119, 6, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(15, 118, 110, 0.18), transparent 24%),
                linear-gradient(180deg, #f6f1e7 0%, #fffdf8 48%, #f5efe4 100%);
            color: #1f2937;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            color: #1c1917;
            letter-spacing: -0.02em;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(28, 25, 23, 0.08);
            border-radius: 24px;
            background: rgba(255, 252, 246, 0.88);
            box-shadow: 0 18px 45px rgba(28, 25, 23, 0.08);
            margin-bottom: 1.2rem;
        }
        .note-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(28, 25, 23, 0.08);
            min-height: 100%;
        }
        .section-title {
            margin-top: 0.6rem;
            margin-bottom: 0.4rem;
            font-size: 1.1rem;
            font-weight: 700;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid rgba(28, 25, 23, 0.12);
            border-radius: 18px;
            padding: 0.85rem 1rem;
            box-shadow: 0 10px 24px rgba(28, 25, 23, 0.06);
        }
        .metric-title {
            color: #9a3412;
            font-weight: 800;
            font-size: 1rem;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            color: #0f172a;
            font-weight: 900;
            font-size: 1.7rem;
            line-height: 1.15;
        }
        div[data-testid="stWidgetLabel"] p,
        .stNumberInput label p,
        .stNumberInput label {
            color: #0f172a !important;
            font-weight: 700 !important;
        }
        .stNumberInput input {
            color: #111827 !important;
            background: rgba(255, 255, 255, 0.96) !important;
        }
        div[data-testid="stFormSubmitButton"] button,
        .stButton button {
            background: linear-gradient(135deg, #0f766e 0%, #115e59 100%) !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 800 !important;
            box-shadow: 0 10px 24px rgba(15, 118, 110, 0.24);
        }
        div[data-testid="stFormSubmitButton"] button:hover,
        .stButton button:hover {
            background: linear-gradient(135deg, #115e59 0%, #134e4a 100%) !important;
            color: #ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

model, metrics = get_artifacts()
metrics = load_metrics_from_disk() if METRICS_PATH.exists() else metrics
dataset = get_dataset()
training_frame = get_training_frame()

st.markdown(
    """
    <div class="hero">
        <h1>Wine Density Prediction</h1>
        <p>
            Interface for the polynomial regression model that predicts wine density
            from 10 physicochemical characteristics of red wine.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-title">R^2</div>
        <div class="metric-value">{metrics['r2']:.4f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
metric_col2.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-title">Accuracy proxy</div>
        <div class="metric-value">{metrics['r2_percent']:.2f}%</div>
    </div>
    """,
    unsafe_allow_html=True,
)
metric_col3.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-title">MAE</div>
        <div class="metric-value">{metrics['mae']:.6f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
metric_col4.markdown(
    f"""
    <div class="metric-card">
        <div class="metric-title">RMSE</div>
        <div class="metric-value">{metrics['rmse']:.6f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.35, 1])

with left_col:
    st.markdown("### Stage 1 Graph")
    st.image(str(PLOT_PATH), caption="Actual density vs predicted density")

with right_col:
    st.markdown("### Model Summary")
    st.markdown(
        f"""
        <div class="note-card">
            <p><strong>Algorithm:</strong> Polynomial Regression (degree 2)</p>
            <p><strong>Target:</strong> {TARGET}</p>
            <p><strong>Features used:</strong> {len(FEATURES)}</p>
            <p><strong>Training rows:</strong> {metrics['train_rows']}</p>
            <p><strong>Test rows:</strong> {metrics['test_rows']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### 10 Dataset Samples")
st.dataframe(dataset.head(10), use_container_width=True)

prep_col, final_data_col = st.columns([0.95, 1.05])

with prep_col:
    st.markdown("### Preprocessing Notes")
    removed_lines = "".join(
        f"<li><strong>{column}</strong>: {reason}</li>"
        for column, reason in EXCLUDED_COLUMNS.items()
    )
    st.markdown(
        f"""
        <div class="note-card">
            <p>The original CSV contains 12 columns. For training, the model uses 10 input features and predicts <strong>{TARGET}</strong>.</p>
            <ul>{removed_lines}</ul>
            <p><strong>{TARGET}</strong> is not used as an input feature because it is the prediction target.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Final columns used for training")
    st.write(training_frame.columns.tolist())

with final_data_col:
    st.markdown("### Final Training Dataset")
    st.dataframe(training_frame.head(10), use_container_width=True)

st.markdown("### Predict Density")
st.write("Enter feature values and click Predict to get the model result.")

medians = dataset[FEATURES].median(numeric_only=True)
mins = dataset[FEATURES].min(numeric_only=True)
maxs = dataset[FEATURES].max(numeric_only=True)

with st.form("prediction_form"):
    input_columns = st.columns(2)
    user_values = {}
    for index, feature in enumerate(FEATURES):
        current_col = input_columns[index % 2]
        with current_col:
            step = 0.01 if float(maxs[feature]) <= 20 else 1.0
            user_values[feature] = st.number_input(
                format_label(feature),
                min_value=float(mins[feature]),
                max_value=float(maxs[feature]),
                value=float(medians[feature]),
                step=step,
            )

    predict_clicked = st.form_submit_button("Predict")

if predict_clicked:
    prediction_model = load_model_from_disk()
    input_frame = pd.DataFrame([user_values], columns=FEATURES)
    prediction = float(prediction_model.predict(input_frame)[0])
    st.success(f"Predicted density: {prediction:.6f}")
    st.dataframe(input_frame, use_container_width=True)

st.caption(f"Files generated near the app: {MODEL_PATH.name}, {PLOT_PATH.name}, {METRICS_PATH.name}")