"""Tests for the LIME Explainer."""

import pytest
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.xai.lime_explainer import LIMEExplainer


@pytest.fixture
def trained_model_and_data():
    """Create and train a Random Forest with test data."""
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def feature_names():
    """Return test feature names."""
    return ['device_trust', 'failed_attempts', 'anomaly_score', 'auth_method', 'location_risk']


class TestLIMEExplainer:
    """Test suite for LIMEExplainer."""
    
    def test_explain_single_returns_dict(self, trained_model_and_data, feature_names):
        """Test that explain_single returns a proper dictionary."""
        model, X, _ = trained_model_and_data
        explainer = LIMEExplainer(X, feature_names=feature_names)
        
        result = explainer.explain_single(X[0], model.predict_proba, num_features=5)
        assert isinstance(result, dict)
        assert 'feature_weights' in result
        assert 'predicted_class' in result
        assert 'latency_ms' in result
    
    def test_feature_weights_present(self, trained_model_and_data, feature_names):
        """Test that feature weights are returned."""
        model, X, _ = trained_model_and_data
        explainer = LIMEExplainer(X, feature_names=feature_names)
        
        result = explainer.explain_single(X[0], model.predict_proba)
        assert len(result['feature_weights']) > 0
    
    def test_text_summary_generated(self, trained_model_and_data, feature_names):
        """Test that a text summary is generated."""
        model, X, _ = trained_model_and_data
        explainer = LIMEExplainer(X, feature_names=feature_names)
        
        result = explainer.explain_single(X[0], model.predict_proba)
        assert 'text_summary' in result
        assert len(result['text_summary']) > 0
    
    def test_prediction_probabilities(self, trained_model_and_data, feature_names):
        """Test that prediction probabilities are included."""
        model, X, _ = trained_model_and_data
        explainer = LIMEExplainer(X, feature_names=feature_names)
        
        result = explainer.explain_single(X[0], model.predict_proba)
        assert 'prediction_probabilities' in result
        assert len(result['prediction_probabilities']) == 2
    
    def test_to_json(self, trained_model_and_data, feature_names):
        """Test JSON serialization."""
        import json
        model, X, _ = trained_model_and_data
        explainer = LIMEExplainer(X, feature_names=feature_names)
        
        result = explainer.explain_single(X[0], model.predict_proba)
        json_str = explainer.to_json(result)
        parsed = json.loads(json_str)
        assert 'feature_weights' in parsed
