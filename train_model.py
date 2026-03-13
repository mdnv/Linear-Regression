import json
import pickle
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "winequality-red.csv"
MODEL_PATH = BASE_DIR / "train_model.pkl"
PLOT_PATH = BASE_DIR / "train_model_plot.png"
METRICS_PATH = BASE_DIR / "model_metrics.json"

FEATURES = [
    "alcohol",
    "fixed acidity",
    "residual sugar",
    "citric acid",
    "pH",
    "chlorides",
    "sulphates",
    "volatile acidity",
    "free sulfur dioxide",
    "total sulfur dioxide",
]
TARGET = "density"
EXCLUDED_COLUMNS = {
    "quality": "removed from training inputs because the model predicts density, not wine quality",
}


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES + [TARGET]].copy()


def train_and_save_artifacts() -> dict:
    df = load_dataset()
    training_df = build_training_frame(df)

    X = training_df[FEATURES]
    y = training_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    x_min, x_max = y_test.min(), y_test.max()
    y_min, y_max = y_pred.min(), y_pred.max()

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue", label="Test samples")
    plt.plot(
        [x_min, x_max],
        [x_min, x_max],
        color="tomato",
        linewidth=2,
        label="Ideal prediction",
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(f"Actual vs Predicted Density | R² = {r2 * 100:.2f}%")
    plt.xlabel("Actual density")
    plt.ylabel("Predicted density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()

    with MODEL_PATH.open("wb") as file_obj:
        pickle.dump(model, file_obj)

    metrics = {
        "r2": float(r2),
        "r2_percent": float(r2 * 100),
        "mae": float(mae),
        "rmse": float(rmse),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "excluded_columns": EXCLUDED_COLUMNS,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "model": model,
        "metrics": metrics,
        "dataset": df,
        "training_df": training_df,
        "plot_path": str(PLOT_PATH),
        "model_path": str(MODEL_PATH),
        "metrics_path": str(METRICS_PATH),
    }


def main() -> int:
    artifacts = train_and_save_artifacts()
    metrics = artifacts["metrics"]

    print(f"R²: {metrics['r2']:.4f} ({metrics['r2_percent']:.2f}%)")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"Graph saved: {PLOT_PATH.name}")
    print(f"Model saved: {MODEL_PATH.name}")
    print(f"Metrics saved: {METRICS_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())