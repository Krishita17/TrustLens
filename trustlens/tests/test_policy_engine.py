"""Tests for the ZTA Policy Engine."""

import pytest
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.zta.policy_engine import ZTAPolicyEngine


@pytest.fixture
def engine():
    """Create a ZTAPolicyEngine instance."""
    return ZTAPolicyEngine(trust_threshold=0.65, reauth_interval_minutes=15)


class TestZTAPolicyEngine:
    """Test suite for ZTAPolicyEngine."""
    
    def test_never_trust_always_verify_valid(self, engine):
        """Test valid request passes verification."""
        request = {
            'user_id': 'user_001',
            'resource_id': 'res_001',
            'timestamp': datetime.now().isoformat(),
        }
        result = engine.never_trust_always_verify(request)
        assert result['verified'] is True
    
    def test_never_trust_always_verify_missing_fields(self, engine):
        """Test request with missing fields fails verification."""
        request = {'user_id': 'user_001'}  # Missing resource_id and timestamp
        result = engine.never_trust_always_verify(request)
        assert result['verified'] is False
    
    def test_least_privilege_allow(self, engine):
        """Test that sufficient clearance allows access."""
        result = engine.least_privilege_check('admin', 3)
        assert result['allowed'] is True
    
    def test_least_privilege_deny(self, engine):
        """Test that insufficient clearance denies access."""
        result = engine.least_privilege_check('viewer', 3)
        assert result['allowed'] is False
    
    def test_continuous_validation_no_session(self, engine):
        """Test that a non-existent session requires re-auth."""
        result = engine.continuous_validation('nonexistent_session')
        assert result['valid'] is False
        assert result['requires_reauth'] is True
    
    def test_continuous_validation_valid_session(self, engine):
        """Test that a fresh session is valid."""
        now = datetime.now()
        engine.refresh_session('session_001', now)
        result = engine.continuous_validation('session_001', now + timedelta(minutes=5))
        assert result['valid'] is True
    
    def test_continuous_validation_expired_session(self, engine):
        """Test that an expired session requires re-auth."""
        now = datetime.now()
        engine.refresh_session('session_002', now)
        result = engine.continuous_validation('session_002', now + timedelta(minutes=20))
        assert result['valid'] is False
        assert result['requires_reauth'] is True
    
    def test_micro_segmentation_allowed(self, engine):
        """Test allowed network segment transition."""
        result = engine.micro_segmentation_check('internal', 'dmz')
        assert result['allowed'] is True
    
    def test_micro_segmentation_denied(self, engine):
        """Test denied network segment transition."""
        result = engine.micro_segmentation_check('public', 'critical')
        assert result['allowed'] is False
    
    def test_compute_trust_score_high(self, engine):
        """Test high trust score results in ALLOW."""
        context = {
            'device_score': 0.9,
            'user_behavior_score': 0.9,
            'network_risk_score': 0.9,
            'auth_method_score': 0.9,
            'location_score': 0.9,
        }
        result = engine.compute_trust_score(context)
        assert result['decision'] == 'ALLOW'
        assert result['trust_score'] >= 0.65
    
    def test_compute_trust_score_low(self, engine):
        """Test low trust score results in DENY."""
        context = {
            'device_score': 0.1,
            'user_behavior_score': 0.1,
            'network_risk_score': 0.1,
            'auth_method_score': 0.1,
            'location_score': 0.1,
        }
        result = engine.compute_trust_score(context)
        assert result['decision'] == 'DENY'
        assert result['trust_score'] < 0.65
    
    def test_compute_trust_score_components(self, engine):
        """Test trust score includes component breakdown."""
        context = {
            'device_score': 0.8,
            'user_behavior_score': 0.7,
            'network_risk_score': 0.6,
            'auth_method_score': 0.9,
            'location_score': 0.5,
        }
        result = engine.compute_trust_score(context)
        assert 'components' in result
        assert len(result['components']) == 5
    
    def test_full_request_evaluation(self, engine):
        """Test complete request evaluation."""
        now = datetime.now()
        engine.refresh_session('session_003', now)
        
        request = {
            'request_id': 'req_001',
            'user_id': 'user_001',
            'resource_id': 'res_001',
            'timestamp': now.isoformat(),
            'user_role': 'admin',
            'resource_sensitivity': 3,
            'session_id': 'session_003',
            'source_segment': 'internal',
            'target_segment': 'corporate',
            'context': {
                'device_score': 0.8,
                'user_behavior_score': 0.8,
                'network_risk_score': 0.7,
                'auth_method_score': 0.9,
                'location_score': 0.8,
            }
        }
        
        result = engine.evaluate_request(request)
        assert result['overall_decision'] == 'ALLOW'
        assert 'checks' in result
