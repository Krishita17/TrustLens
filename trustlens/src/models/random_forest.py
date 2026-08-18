"""
Random Forest classifier for ZTA authentication decisions.

Primary model — best suited for SHAP TreeExplainer.
"""

import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


def build_random_forest(n_estimators: int = 200, max_depth: int = 15,
                        random_state: int = 42, **kwargs) -> RandomForestClassifier:
    """
    Build a Random Forest classifier with ZTA-optimized hyperparameters.
    
    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree.
        random_state: Random seed.
        **kwargs: Additional hyperparameters.
        
    Returns:
        Configured (untrained) RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
        min_samples_split=kwargs.get('min_samples_split', 5),
        min_samples_leaf=kwargs.get('min_samples_leaf', 2),
    )
    logger.info(f"Built RandomForest: {n_estimators} estimators, max_depth={max_depth}")
    return model


def train_random_forest(model: RandomForestClassifier, X_train: np.ndarray,
                        y_train: np.ndarray, cv_folds: int = 5) -> dict:
    """
    Train Random Forest with cross-validation.
    
    Args:
        model: Untrained RandomForestClassifier.
        X_train: Training features.
        y_train: Training labels.
        cv_folds: Number of cross-validation folds.
        
    Returns:
        Dictionary with trained model and CV scores.
    """
    logger.info(f"Training Random Forest with {cv_folds}-fold CV...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1_weighted')
    logger.info(f"CV F1 scores: {cv_scores.round(4)} (mean: {cv_scores.mean():.4f})")
    
    # Fit on full training set
    model.fit(X_train, y_train)
    logger.info("Random Forest training complete")
    
    return {
        'model': model,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importances': model.feature_importances_
    }
