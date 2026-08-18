"""Tests for the SHAP Explainer."""

import pytest
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.xai.shap_explainer import SHAPExplainer


@pytest.fixture
def trained_rf_model():
    """Create and train a simple Random Forest for testing."""
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X


@pytest.fixture
def feature_names():
    """Return test feature names."""
    return ['device_trust', 'failed_attempts', 'anomaly_score', 'auth_method', 'location_risk']


class TestSHAPExplainer:
    """Test suite for SHAPExplainer."""
    
    def test_explain_single_returns_dict(self, trained_rf_model, feature_names):
        """Test that explain_single returns a proper dictionary."""
        model, X = trained_rf_model
        explainer = SHAPExplainer(model, model_type='tree', feature_names=feature_names)
        
        result = explainer.explain_single(X[0])
        assert isinstance(result, dict)
        assert 'shap_values' in result
        assert 'base_value' in result
        assert 'feature_names' in result
        assert 'top_features' in result
        assert 'latency_ms' in result
    
    def test_shap_values_length_matches_features(self, trained_rf_model, feature_names):
        """Test that SHAP values have same length as features."""
        model, X = trained_rf_model
        explainer = SHAPExplainer(model, model_type='tree', feature_names=feature_names)
        
        result = explainer.explain_single(X[0])
        assert len(result['shap_values']) == len(feature_names)
    
    def test_explain_batch(self, trained_rf_model, feature_names):
        """Test batch SHAP value computation."""
        model, X = trained_rf_model
        explainer = SHAPExplainer(model, model_type='tree', feature_names=feature_names)
        
        shap_values = explainer.explain_batch(X[:10])
        assert shap_values.shape == (10, 5)
    
    def test_global_importance(self, trained_rf_model, feature_names):
        """Test global feature importance computation."""
        model, X = trained_rf_model
        explainer = SHAPExplainer(model, model_type='tree', feature_names=feature_names)
        
        importance = explainer.get_global_importance(X[:50])
        assert len(importance) == len(feature_names)
        assert all(v >= 0 for v in importance.values())
    
    def test_to_json(self, trained_rf_model, feature_names):
        """Test JSON serialization of explanation."""
        import json
        model, X = trained_rf_model
        explainer = SHAPExplainer(model, model_type='tree', feature_names=feature_names)
        
        result = explainer.explain_single(X[0])
        json_str = explainer.to_json(result)
        parsed = json.loads(json_str)
        assert 'shap_values' in parsed
    
    def test_top_features_sorted_by_importance(self, trained_rf_model, feature_names):
        """Test that top features are sorted by absolute SHAP value."""
        model, X = trained_rf_model
        explainer = SHAPExplainer(model, model_type='tree', feature_names=feature_names)
        
        result = explainer.explain_single(X[0])
        top = result['top_features']
        abs_values = [abs(f['shap_value']) for f in top]
        assert abs_values == sorted(abs_values, reverse=True)
