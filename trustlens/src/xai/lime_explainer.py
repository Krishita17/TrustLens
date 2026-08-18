"""
LIME Explainer for TrustLens authentication decisions.

Uses LimeTabularExplainer for local, model-agnostic explanations
of individual authentication decisions.
"""

import logging
import time
import json
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """LIME-based explanation generator for ZTA authentication decisions."""
    
    def __init__(self, training_data: np.ndarray, feature_names: list = None,
                 class_names: list = None, mode: str = 'classification'):
        """
        Initialize LIME explainer.
        
        Args:
            training_data: Training dataset for perturbation reference.
            feature_names: List of feature names.
            class_names: List of class names (e.g., ['ALLOW', 'DENY']).
            mode: 'classification' or 'regression'.
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(training_data.shape[1])]
        self.class_names = class_names or ['ALLOW', 'DENY']
        
        self.explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode,
            discretize_continuous=True,
            random_state=42
        )
        
        logger.info(f"Initialized LIME explainer with {training_data.shape[1]} features")
    
    def explain_single(self, instance: np.ndarray, predict_fn, num_features: int = 10,
                       num_samples: int = 1000) -> dict:
        """
        Generate LIME explanation for a single authentication decision.
        
        Args:
            instance: Single feature vector (1D array).
            predict_fn: Model's predict_proba function.
            num_features: Number of top features to include.
            num_samples: Number of perturbation samples.
            
        Returns:
            Dictionary with LIME explanation data.
        """
        start_time = time.perf_counter()
        
        if instance.ndim > 1:
            instance = instance.flatten()
        
        explanation = self.explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=1
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Extract explanation details
        label = explanation.available_labels()[0]
        feature_weights = explanation.as_list(label=label)
        
        result = {
            'predicted_class': int(label),
            'predicted_class_name': self.class_names[label] if label < len(self.class_names) else str(label),
            'prediction_probabilities': explanation.predict_proba.tolist(),
            'feature_weights': [
                {'feature': feat, 'weight': round(weight, 4)}
                for feat, weight in feature_weights
            ],
            'intercept': float(explanation.intercept[label]),
            'score': float(explanation.score),
            'latency_ms': round(latency_ms, 2),
            'num_samples': num_samples,
        }
        
        # Generate text summary
        result['text_summary'] = self._generate_text_summary(result)
        
        return result
    
    def _generate_text_summary(self, explanation: dict) -> str:
        """
        Generate human-readable text summary of the explanation.
        
        Args:
            explanation: Explanation dictionary.
            
        Returns:
            Text summary string.
        """
        decision = explanation['predicted_class_name']
        features = explanation['feature_weights'][:5]  # Top 5
        
        summary = f"Decision: {decision}\n"
        summary += f"Confidence: {max(explanation['prediction_probabilities']):.1%}\n"
        summary += "Top contributing factors:\n"
        
        for i, fw in enumerate(features, 1):
            direction = "increases" if fw['weight'] > 0 else "decreases"
            summary += f"  {i}. {fw['feature']} ({direction} risk by {abs(fw['weight']):.3f})\n"
        
        return summary
    
    def explain_batch(self, X: np.ndarray, predict_fn, num_features: int = 10,
                      num_samples: int = 1000) -> list:
        """
        Generate LIME explanations for multiple instances.
        
        Args:
            X: Feature matrix (2D array).
            predict_fn: Model's predict_proba function.
            num_features: Number of top features per explanation.
            num_samples: Perturbation samples per instance.
            
        Returns:
            List of explanation dictionaries.
        """
        explanations = []
        for i in range(len(X)):
            exp = self.explain_single(X[i], predict_fn, num_features, num_samples)
            explanations.append(exp)
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{len(X)} LIME explanations")
        
        return explanations
    
    def to_json(self, explanation: dict) -> str:
        """Serialize explanation to JSON string."""
        return json.dumps(explanation, indent=2)
