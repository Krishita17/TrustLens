"""Tests for the TrustScorer."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.zta.trust_scorer import TrustScorer


@pytest.fixture
def scorer():
    """Create a TrustScorer instance."""
    return TrustScorer(threshold=0.65)


class TestTrustScorer:
    """Test suite for TrustScorer."""
    
    def test_high_trust_allows(self, scorer):
        """Test that a high-trust context results in ALLOW."""
        context = {
            'device_trust_score': 0.9,
            'patch_level': 5,
            'vpn_active': True,
            'failed_attempts': 0,
            'anomaly_score': 0.1,
            'auth_method': 'biometric',
            'location_risk': 'low',
        }
        result = scorer.compute_trust_score(context)
        assert result['decision'] == 'ALLOW'
        assert result['trust_score'] >= 0.65
    
    def test_low_trust_denies(self, scorer):
        """Test that a low-trust context results in DENY."""
        context = {
            'device_trust_score': 0.1,
            'patch_level': 200,
            'vpn_active': False,
            'failed_attempts': 10,
            'anomaly_score': 0.9,
            'auth_method': 'password',
            'location_risk': 'high',
        }
        result = scorer.compute_trust_score(context)
        assert result['decision'] == 'DENY'
        assert result['trust_score'] < 0.65
    
    def test_trust_score_in_range(self, scorer):
        """Test that trust score is always between 0 and 1."""
        context = {
            'device_trust_score': 0.5,
            'patch_level': 30,
            'vpn_active': True,
            'failed_attempts': 2,
            'anomaly_score': 0.3,
            'auth_method': 'MFA',
            'location_risk': 'medium',
        }
        result = scorer.compute_trust_score(context)
        assert 0.0 <= result['trust_score'] <= 1.0
    
    def test_remediation_suggestions_for_denied(self, scorer):
        """Test that remediation suggestions are provided for denied requests."""
        context = {
            'device_trust_score': 0.2,
            'patch_level': 100,
            'vpn_active': False,
            'failed_attempts': 5,
            'anomaly_score': 0.8,
            'auth_method': 'password',
            'location_risk': 'high',
        }
        result = scorer.compute_trust_score(context)
        suggestions = scorer.get_remediation_suggestions(result)
        assert len(suggestions) > 0
        assert all('action' in s for s in suggestions)
    
    def test_no_remediation_for_allowed(self, scorer):
        """Test no remediation suggestions when request is allowed."""
        context = {
            'device_trust_score': 0.9,
            'patch_level': 5,
            'vpn_active': True,
            'failed_attempts': 0,
            'anomaly_score': 0.1,
            'auth_method': 'biometric',
            'location_risk': 'low',
        }
        result = scorer.compute_trust_score(context)
        suggestions = scorer.get_remediation_suggestions(result)
        assert len(suggestions) == 0
    
    def test_auth_method_scoring(self, scorer):
        """Test different auth methods produce different scores."""
        password_score = scorer.compute_auth_method_score('password')
        mfa_score = scorer.compute_auth_method_score('MFA')
        bio_score = scorer.compute_auth_method_score('biometric')
        
        assert password_score < mfa_score < bio_score
    
    def test_location_scoring(self, scorer):
        """Test different locations produce appropriate scores."""
        low_score = scorer.compute_location_score('low')
        high_score = scorer.compute_location_score('high')
        
        assert low_score > high_score
    
    def test_weights_sum_to_one(self, scorer):
        """Test that weights are normalized to sum to 1."""
        total = sum(scorer.weights.values())
        assert abs(total - 1.0) < 0.01
    
    def test_component_breakdown(self, scorer):
        """Test that trust result contains component breakdown."""
        context = {
            'device_trust_score': 0.7,
            'patch_level': 20,
            'vpn_active': True,
            'failed_attempts': 1,
            'anomaly_score': 0.2,
            'auth_method': 'MFA',
            'location_risk': 'low',
        }
        result = scorer.compute_trust_score(context)
        assert 'components' in result
        assert len(result['components']) == 5
        for comp_name, comp_data in result['components'].items():
            assert 'score' in comp_data
            assert 'weight' in comp_data
            assert 'contribution' in comp_data
