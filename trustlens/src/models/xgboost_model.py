"""
XGBoost classifier for ZTA authentication decisions.

Secondary model with gradient boosting for comparison.
"""

import logging
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


def build_xgboost(
    learning_rate: float = 0.05,
    max_depth: int = 6,
    n_estimators: int = 300,
    random_state: int = 42,
    scale_pos_weight: float = None,
    **kwargs
) -> XGBClassifier:
    """
    Build an XGBoost classifier with ZTA-optimized hyperparameters.
    """

    model = XGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight if scale_pos_weight is not None else 1.0,
        random_state=random_state,
        eval_metric="logloss",
        subsample=kwargs.get("subsample", 0.8),
        colsample_bytree=kwargs.get("colsample_bytree", 0.8),
        n_jobs=-1
    )

    logger.info(
        f"Built XGBoost: lr={learning_rate}, depth={max_depth}, n_est={n_estimators}"
    )

    return model


def train_xgboost(
    model: XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    cv_folds: int = 5
) -> dict:
    """
    Train XGBoost model with cross-validation and optional validation set.
    """

    logger.info(f"Training XGBoost with {cv_folds}-fold CV...")

    # -----------------------------
    # FIXED: safer imbalance handling
    # -----------------------------
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)

    if pos_count > 0:
        scale_weight = neg_count / pos_count
        model.set_params(scale_pos_weight=scale_weight)
        logger.info(f"Auto-set scale_pos_weight: {scale_weight:.2f}")

    # -----------------------------
    # Cross-validation
    # -----------------------------
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv_folds,
        scoring="f1_weighted"
    )

    logger.info(
        f"CV F1 scores: {np.round(cv_scores, 4)} "
        f"(mean: {cv_scores.mean():.4f})"
    )

    # -----------------------------
    # Fit model
    # -----------------------------
    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)

    logger.info("XGBoost training complete")

    return {
        "model": model,
        "cv_scores": cv_scores,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "feature_importances": getattr(model, "feature_importances_", None)
    }
