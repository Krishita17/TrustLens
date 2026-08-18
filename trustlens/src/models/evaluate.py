"""
Model evaluation for TrustLens authentication classifiers.

Computes accuracy, precision, recall, F1, AUC-ROC, and confusion matrix.
Also measures inference time for real-time ZTA feasibility assessment.
"""

import os
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

logger = logging.getLogger(__name__)


def safe_float(val):
    """Safely format float values for logging."""
    return f"{val:.4f}" if val is not None else "N/A"


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "model") -> dict:
    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
    except (AttributeError, IndexError, ValueError):
        y_proba = None
        auc_roc = None
        fpr, tpr = None, None

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "auc_roc": auc_roc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    if fpr is not None and tpr is not None:
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

    inference_time = measure_inference_time(model, X_test)
    metrics["inference_time_ms"] = inference_time

    logger.info(
        f"{model_name} — Accuracy: {metrics['accuracy']:.4f}, "
        f"F1: {safe_float(metrics['f1_score'])}, "
        f"AUC-ROC: {safe_float(auc_roc)}, "
        f"Inference: {inference_time:.2f}ms"
    )

    return metrics


def measure_inference_time(model, X_test: np.ndarray, n_runs: int = 100) -> float:
    """Measure average inference time per prediction."""
    single_sample = X_test[:1] if hasattr(X_test, "__getitem__") else X_test

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(single_sample)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return round(float(np.mean(times)), 4)


def compare_models(results: list) -> pd.DataFrame:
    """Compare multiple model evaluation results."""
    comparison_cols = [
        "model_name",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc_roc",
        "inference_time_ms",
    ]

    rows = []
    for r in results:
        rows.append({col: r.get(col) for col in comparison_cols})

    df = pd.DataFrame(rows)
    logger.info(f"\nModel Comparison:\n{df.to_string(index=False)}")
    return df


def save_metrics(metrics, output_path: str):
    """
    Save one metrics dict or a list of metrics dicts to CSV.
    """
    rows = []

    if isinstance(metrics, list):
        for item in metrics:
            rows.append(
                {
                    "model_name": item["model_name"],
                    "accuracy": item["accuracy"],
                    "precision": item["precision"],
                    "recall": item["recall"],
                    "f1_score": item["f1_score"],
                    "auc_roc": item.get("auc_roc", ""),
                    "inference_time_ms": item["inference_time_ms"],
                    "timestamp": datetime.now().isoformat(),
                }
            )
    else:
        rows.append(
            {
                "model_name": metrics["model_name"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "auc_roc": metrics.get("auc_roc", ""),
                "inference_time_ms": metrics["inference_time_ms"],
                "timestamp": datetime.now().isoformat(),
            }
        )

    df = pd.DataFrame(rows)

    if os.path.exists(output_path):
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df.to_csv(output_path, index=False)

    logger.info(f"Metrics saved to {output_path}")
