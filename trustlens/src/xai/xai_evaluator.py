"""
XAI Evaluation Metrics for TrustLens research.

Novel contribution: comprehensive evaluation framework for comparing
explanation quality across SHAP, LIME, and Anchor methods.

Metrics:
- Faithfulness: Does removing top features degrade model performance?
- Stability: Same input → same explanation across runs?
- Sparsity: How many features needed to explain decision?
- Comprehensibility: Is explanation readable by non-ML analysts?
- Latency: Time to generate explanation (must be < 500ms for real-time).
"""

import time
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class XAIEvaluator:
    """Evaluate and compare XAI explanation methods."""
    
    def __init__(self, model, feature_names: list = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model with predict() and predict_proba().
            feature_names: List of feature names.
        """
        self.model = model
        self.feature_names = feature_names
    
    def evaluate_faithfulness(self, X: np.ndarray, y: np.ndarray,
                              feature_importances: np.ndarray,
                              top_k_values: list = None) -> dict:
        """
        Measure explanation faithfulness by feature removal.
        
        Removes top-k important features and measures performance degradation.
        Higher degradation = more faithful explanation.
        
        Args:
            X: Test feature matrix.
            y: True labels.
            feature_importances: Feature importance scores (from SHAP/LIME).
            top_k_values: List of k values to test. Default: [1, 3, 5, 10].
            
        Returns:
            Dictionary with faithfulness scores per k.
        """
        from sklearn.metrics import accuracy_score
        
        if top_k_values is None:
            top_k_values = [1, 3, 5, 10]
        
        # Baseline performance
        baseline_acc = accuracy_score(y, self.model.predict(X))
        
        # Rank features by importance
        importance_ranking = np.argsort(-np.abs(feature_importances))
        
        results = {'baseline_accuracy': baseline_acc, 'degradation': {}}
        
        for k in top_k_values:
            if k > X.shape[1]:
                continue
            
            # Zero out top-k features
            X_modified = X.copy()
            top_features = importance_ranking[:k]
            X_modified[:, top_features] = 0
            
            modified_acc = accuracy_score(y, self.model.predict(X_modified))
            degradation = baseline_acc - modified_acc
            
            results['degradation'][k] = {
                'accuracy_after_removal': round(modified_acc, 4),
                'degradation': round(degradation, 4),
                'features_removed': top_features.tolist()
            }
        
        # Faithfulness score: average normalized degradation
        degradations = [v['degradation'] for v in results['degradation'].values()]
        results['faithfulness_score'] = round(
            np.mean(degradations) / max(baseline_acc, 1e-10), 4
        ) if degradations else 0.0
        
        logger.info(f"Faithfulness score: {results['faithfulness_score']}")
        return results
    
    def evaluate_stability(self, instance: np.ndarray, explain_fn,
                           n_runs: int = 10) -> dict:
        """
        Measure explanation stability across multiple runs.
        
        Same input should produce same (or very similar) explanation.
        
        Args:
            instance: Single feature vector.
            explain_fn: Function that returns explanation (feature importances).
            n_runs: Number of repeated explanation runs.
            
        Returns:
            Dictionary with stability metrics.
        """
        explanations = []
        
        for _ in range(n_runs):
            exp = explain_fn(instance)
            if isinstance(exp, dict) and 'shap_values' in exp:
                explanations.append(np.array(exp['shap_values']))
            elif isinstance(exp, np.ndarray):
                explanations.append(exp)
            else:
                explanations.append(np.array(exp))
        
        explanations = np.array(explanations)
        
        # Compute pairwise cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = []
        for i in range(len(explanations)):
            for j in range(i + 1, len(explanations)):
                sim = cosine_similarity(
                    explanations[i].reshape(1, -1),
                    explanations[j].reshape(1, -1)
                )[0][0]
                similarities.append(sim)
        
        result = {
            'mean_similarity': round(float(np.mean(similarities)), 4) if similarities else 1.0,
            'min_similarity': round(float(np.min(similarities)), 4) if similarities else 1.0,
            'std_similarity': round(float(np.std(similarities)), 4) if similarities else 0.0,
            'n_runs': n_runs,
            'is_stable': float(np.mean(similarities)) > 0.95 if similarities else True
        }
        
        logger.info(f"Stability: mean_similarity={result['mean_similarity']}")
        return result
    
    def evaluate_sparsity(self, explanation_values: np.ndarray,
                          threshold: float = 0.01) -> dict:
        """
        Measure explanation sparsity.
        
        Fewer features needed = more sparse = more comprehensible.
        
        Args:
            explanation_values: Feature importance/SHAP values.
            threshold: Minimum absolute value to count as significant.
            
        Returns:
            Dictionary with sparsity metrics.
        """
        abs_values = np.abs(np.array(explanation_values))
        total_features = len(abs_values)
        significant_features = int(np.sum(abs_values > threshold))
        
        result = {
            'total_features': total_features,
            'significant_features': significant_features,
            'sparsity_ratio': round(1 - significant_features / max(total_features, 1), 4),
            'threshold': threshold,
        }
        
        logger.info(f"Sparsity: {significant_features}/{total_features} significant features")
        return result
    
    def evaluate_latency(self, explain_fn, instance: np.ndarray,
                         n_runs: int = 50) -> dict:
        """
        Measure explanation generation latency.
        
        Must be under 500ms for real-time ZTA.
        
        Args:
            explain_fn: Function that generates explanation for an instance.
            instance: Single feature vector.
            n_runs: Number of timed runs.
            
        Returns:
            Dictionary with latency statistics.
        """
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            explain_fn(instance)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        
        result = {
            'mean_ms': round(float(np.mean(times)), 2),
            'median_ms': round(float(np.median(times)), 2),
            'p95_ms': round(float(np.percentile(times, 95)), 2),
            'max_ms': round(float(np.max(times)), 2),
            'min_ms': round(float(np.min(times)), 2),
            'meets_realtime': float(np.percentile(times, 95)) < 500,
            'n_runs': n_runs,
        }
        
        logger.info(f"Latency: mean={result['mean_ms']}ms, p95={result['p95_ms']}ms, "
                     f"realtime={'YES' if result['meets_realtime'] else 'NO'}")
        return result
    
    def full_evaluation(self, X_test: np.ndarray, y_test: np.ndarray,
                        explain_fn, feature_importances: np.ndarray,
                        method_name: str = "XAI") -> dict:
        """
        Run complete XAI evaluation suite.
        
        Args:
            X_test: Test features.
            y_test: True labels.
            explain_fn: Explanation function for single instances.
            feature_importances: Global feature importance scores.
            method_name: Name of XAI method being evaluated.
            
        Returns:
            Comprehensive evaluation results dictionary.
        """
        logger.info(f"Running full XAI evaluation for {method_name}...")
        
        results = {
            'method': method_name,
            'timestamp': datetime.now().isoformat(),
            'faithfulness': self.evaluate_faithfulness(X_test, y_test, feature_importances),
            'stability': self.evaluate_stability(X_test[0], explain_fn),
            'sparsity': self.evaluate_sparsity(feature_importances),
            'latency': self.evaluate_latency(explain_fn, X_test[0]),
        }
        
        logger.info(f"Full evaluation complete for {method_name}")
        return results
    
    def save_metrics(self, results: dict, output_path: str):
        """
        Save evaluation metrics to CSV.
        
        Args:
            results: Evaluation results dictionary.
            output_path: Path to output CSV file.
        """
        import pandas as pd
        import os
        
        row = {
            'xai_method': results['method'],
            'model_name': 'model',
            'faithfulness': results['faithfulness']['faithfulness_score'],
            'stability': results['stability']['mean_similarity'],
            'sparsity': results['sparsity']['sparsity_ratio'],
            'comprehensibility': 1.0 if results['sparsity']['significant_features'] <= 10 else 0.5,
            'latency_ms': results['latency']['mean_ms'],
            'timestamp': results['timestamp']
        }
        
        df = pd.DataFrame([row])
        if os.path.exists(output_path):
            df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_path, index=False)
        
        logger.info(f"XAI metrics saved to {output_path}")
