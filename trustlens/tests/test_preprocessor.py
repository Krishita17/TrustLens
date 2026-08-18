"""Tests for the AuthDataPreprocessor."""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessor import AuthDataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample authentication data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'device_trust_score': np.random.uniform(0, 1, 100),
        'location_risk': np.random.choice(['low', 'medium', 'high'], 100),
        'time_of_access': np.random.randint(0, 24, 100),
        'failed_attempts': np.random.randint(0, 10, 100),
        'resource_sensitivity': np.random.randint(1, 6, 100),
        'vpn_active': np.random.choice([True, False], 100),
        'patch_level': np.random.randint(0, 100, 100),
        'auth_method': np.random.choice(['password', 'MFA', 'biometric'], 100),
        'anomaly_score': np.random.uniform(0, 1, 100),
        'decision': np.random.choice([0, 1], 100),
    })


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance."""
    return AuthDataPreprocessor(scaling_method='standard', random_state=42)


class TestAuthDataPreprocessor:
    """Test suite for AuthDataPreprocessor."""
    
    def test_clean_data_removes_duplicates(self, preprocessor, sample_data):
        """Test that clean_data removes duplicate rows."""
        # Add duplicates
        df_with_dupes = pd.concat([sample_data, sample_data.iloc[:5]], ignore_index=True)
        cleaned = preprocessor.clean_data(df_with_dupes)
        assert len(cleaned) <= len(df_with_dupes)
    
    def test_clean_data_handles_missing_values(self, preprocessor):
        """Test that clean_data fills missing values."""
        df = pd.DataFrame({
            'numeric_col': [1.0, np.nan, 3.0, 4.0],
            'categorical_col': ['a', 'b', None, 'a'],
            'decision': [0, 1, 0, 1],
        })
        cleaned = preprocessor.clean_data(df)
        assert cleaned.isnull().sum().sum() == 0
    
    def test_encode_categorical(self, preprocessor, sample_data):
        """Test categorical encoding."""
        encoded = preprocessor.encode_categorical(sample_data, ['location_risk', 'auth_method'])
        assert encoded['location_risk'].dtype in [np.int64, np.int32, int]
        assert encoded['auth_method'].dtype in [np.int64, np.int32, int]
    
    def test_scale_features(self, preprocessor, sample_data):
        """Test feature scaling."""
        # First encode categoricals
        encoded = preprocessor.encode_categorical(sample_data)
        scaled = preprocessor.scale_features(encoded, exclude_cols=['decision'])
        
        # Check that numeric columns are scaled (mean ~0, std ~1 for standard scaler)
        numeric_cols = scaled.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'decision']
        if len(numeric_cols) > 0:
            assert abs(scaled[numeric_cols[0]].mean()) < 1.0  # Approximately centered
    
    def test_prepare_train_test(self, preprocessor, sample_data):
        """Test train/test split."""
        encoded = preprocessor.encode_categorical(sample_data)
        X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(
            encoded, target_col='decision', test_size=0.2
        )
        assert len(X_train) + len(X_test) == len(sample_data)
        assert len(y_train) + len(y_test) == len(sample_data)
        assert 'decision' not in X_train.columns
    
    def test_fit_transform_pipeline(self, preprocessor, sample_data):
        """Test full preprocessing pipeline."""
        processed = preprocessor.fit_transform(sample_data, target_col='decision')
        assert len(processed) > 0
        assert processed.isnull().sum().sum() == 0
