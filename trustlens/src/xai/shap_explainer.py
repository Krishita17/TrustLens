"""
SHAP Explainer for TrustLens authentication decisions.

Implements TreeExplainer for Random Forest and XGBoost,
and KernelExplainer for Neural Network models.
Generates waterfall, force, summary, and dependency plots.
"""

import logging
import json
import time
import numpy as np
import shap

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based explanation generator for ZTA authentication decisions."""
    
    def __init__(self, model, model_type: str = 'tree', feature_names: list = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (sklearn, xgboost, or neural net).
            model_type: 'tree' for RF/XGBoost, 'kernel' for neural net.
            feature_names: List of feature names for readable explanations.
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.explainer = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Initialize the appropriate SHAP explainer based on model type."""
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Initialized SHAP TreeExplainer")
        elif self.model_type == 'kernel':
            # KernelExplainer needs a prediction function and background data
            logger.info("KernelExplainer will be initialized when explain() is called with background data")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    @staticmethod
    def _extract_class1_values(shap_values):
        """
        Extract SHAP values for class 1 (DENY) from various SHAP output formats.

        Handles both older list-of-arrays format and newer 3D ndarray format.

        Args:
            shap_values: Raw output from explainer.shap_values().

        Returns:
            numpy array of SHAP values for class 1.
        """
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Shape: (n_samples, n_features, n_classes) — pick class 1
            sv = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
        else:
            sv = shap_values
        return sv

    def explain_single(self, instance: np.ndarray, background_data: np.ndarray = None) -> dict:
        """
        Generate SHAP explanation for a single authentication decision.
        
        Args:
            instance: Single feature vector (1D or 2D array).
            background_data: Background dataset for KernelExplainer.
            
        Returns:
            Dictionary with SHAP values and explanation metadata.
        """
        start_time = time.perf_counter()
        
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)
        
        if self.model_type == 'kernel' and self.explainer is None:
            if background_data is None:
                raise ValueError("KernelExplainer requires background_data")
            predict_fn = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict
            self.explainer = shap.KernelExplainer(predict_fn, background_data)
        
        shap_values = self.explainer.shap_values(instance)
        
        # Extract class 1 (DENY) SHAP values
        sv = self._extract_class1_values(shap_values)
        
        # Reduce to single instance
        if sv.ndim > 1:
            sv = sv[0]
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Get base value
        expected = self.explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            base_val = float(expected[1]) if len(expected) > 1 else float(expected[0])
        else:
            base_val = float(expected)
        
        # Build explanation object
        sv_list = sv.tolist()
        explanation = {
            'shap_values': sv_list,
            'base_value': base_val,
            'feature_names': self.feature_names or [f'feature_{i}' for i in range(len(sv))],
            'feature_values': instance[0].tolist(),
            'latency_ms': round(latency_ms, 2),
        }
        
        # Top contributing features
        feature_contributions = sorted(
            zip(explanation['feature_names'], sv_list, instance[0].tolist()),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        explanation['top_features'] = [
            {'feature': f, 'shap_value': round(v, 4), 'feature_value': round(fv, 4)}
            for f, v, fv in feature_contributions[:10]
        ]
        
        return explanation
    
    def explain_batch(self, X: np.ndarray, background_data: np.ndarray = None) -> np.ndarray:
        """
        Compute SHAP values for a batch of instances.
        
        Args:
            X: Feature matrix (2D array).
            background_data: Background data for KernelExplainer.
            
        Returns:
            Array of SHAP values with shape (n_samples, n_features).
        """
        if self.model_type == 'kernel' and self.explainer is None:
            if background_data is None:
                raise ValueError("KernelExplainer requires background_data")
            predict_fn = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict
            self.explainer = shap.KernelExplainer(predict_fn, background_data)
        
        shap_values = self.explainer.shap_values(X)
        return self._extract_class1_values(shap_values)
    
    def get_global_importance(self, X: np.ndarray, background_data: np.ndarray = None) -> dict:
        """
        Compute global feature importance using mean absolute SHAP values.
        
        Args:
            X: Feature matrix for computing global importance.
            background_data: Background data for KernelExplainer.
            
        Returns:
            Dictionary mapping feature names to importance scores.
        """
        shap_values = self.explain_batch(X, background_data)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        names = self.feature_names or [f'feature_{i}' for i in range(len(mean_abs_shap))]
        importance = dict(sorted(
            zip(names, mean_abs_shap.tolist()),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return importance
    
    def to_json(self, explanation: dict) -> str:
        """
        Serialize explanation to JSON string.
        
        Args:
            explanation: Explanation dictionary from explain_single().
            
        Returns:
            JSON string.
        """
        return json.dumps(explanation, indent=2)
