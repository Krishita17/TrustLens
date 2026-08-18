"""
Training pipeline for all TrustLens authentication models.
"""
import os
import sys
import logging
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.synthetic_generator import generate_synthetic_auth_data
from src.data.preprocessor import AuthDataPreprocessor
from src.data.feature_engineering import ZTAFeatureEngineer
from src.models.random_forest import build_random_forest, train_random_forest
from src.models.xgboost_model import build_xgboost, train_xgboost
from src.models.neural_net import NeuralNetClassifier
from src.models.evaluate import evaluate_model, compare_models, save_metrics
from src.models.model_utils import save_model

logger = logging.getLogger(__name__)


def prepare_data(data_path: str = None) -> tuple:
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}: {df.shape}")
    else:
        logger.info("Generating synthetic data for training...")
        df = generate_synthetic_auth_data(n_samples=50000)

    engineer = ZTAFeatureEngineer()
    df = engineer.build_all_features(df)

    drop_cols = ["user_id", "decision"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    target_col = "decision_binary"
    preprocessor = AuthDataPreprocessor(scaling_method="standard")
    df = preprocessor.clean_data(df)
    df = preprocessor.encode_categorical(df)

    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(
        df,
        target_col=target_col,
        test_size=0.2,
    )

    feature_names = X_train.columns.tolist()
    X_train_arr = preprocessor.scaler.fit_transform(X_train)
    X_test_arr = preprocessor.scaler.transform(X_test)

    return X_train_arr, X_test_arr, y_train.values, y_test.values, feature_names


def train_all_models(X_train, X_test, y_train, y_test, feature_names):
    results = {}

    logger.info("=" * 60)
    logger.info("Training Random Forest...")
    logger.info("=" * 60)
    rf_model = build_random_forest(n_estimators=200, max_depth=15)
    rf_result = train_random_forest(rf_model, X_train, y_train, cv_folds=5)
    rf_metrics = evaluate_model(rf_result["model"], X_test, y_test, "RandomForest")
    results["random_forest"] = {
        "model": rf_result["model"],
        "metrics": rf_metrics,
        "cv_scores": rf_result["cv_scores"],
    }

    logger.info("=" * 60)
    logger.info("Training XGBoost...")
    logger.info("=" * 60)
    xgb_model = build_xgboost(learning_rate=0.05, max_depth=6, n_estimators=300)
    xgb_result = train_xgboost(xgb_model, X_train, y_train, cv_folds=5)
    xgb_metrics = evaluate_model(xgb_result["model"], X_test, y_test, "XGBoost")
    results["xgboost"] = {
        "model": xgb_result["model"],
        "metrics": xgb_metrics,
        "cv_scores": xgb_result["cv_scores"],
    }

    logger.info("=" * 60)
    logger.info("Training Neural Network...")
    logger.info("=" * 60)
    nn_model = NeuralNetClassifier(
        input_dim=X_train.shape[1],
        hidden_layers=[128, 64, 32],
        dropout=0.3,
        lr=0.001,
        epochs=100,
        batch_size=1024,
        patience=10,       # ← explicitly set
        random_state=42,   # ← explicitly set
        verbose=True,      # ← now accepted
    )
    nn_model.fit(X_train, y_train)
    nn_metrics = evaluate_model(nn_model, X_test, y_test, "NeuralNetwork")
    results["neural_network"] = {
        "model": nn_model,
        "metrics": nn_metrics,
    }

    comparison = compare_models([rf_metrics, xgb_metrics, nn_metrics])

    model_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    os.makedirs(model_dir, exist_ok=True)

    for name, data in results.items():
        save_model(data["model"], name, model_dir)

    metrics_path = os.path.join(model_dir, "model_metrics.csv")
    all_metrics = [data["metrics"] for data in results.values()]
    save_metrics(all_metrics, metrics_path)

    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(f"Models saved to {model_dir}")

    return results, comparison


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )

    print("=" * 60)
    print("TrustLens Model Training Pipeline")
    print("=" * 60)

    data_path = os.path.join(PROJECT_ROOT, "data", "synthetic", "generated_auth_logs.csv")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(data_path)
    print(f"Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test")

    results, comparison = train_all_models(
        X_train, X_test, y_train, y_test, feature_names
    )

    print("\nModel Comparison:")
    print(comparison.to_string(index=False))
