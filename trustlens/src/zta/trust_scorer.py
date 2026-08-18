"""
Trust Score Calculator for TrustLens.

Computes a continuous trust score for each authentication request
based on device, user behavior, network, auth method, and location factors.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class TrustScorer:
    """Calculate trust scores for Zero Trust authentication decisions."""
    
    # Default weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        'device_score': 0.30,
        'user_behavior_score': 0.25,
        'network_risk_score': 0.20,
        'auth_method_score': 0.15,
        'location_score': 0.10,
    }
    
    # Auth method strength mappings
    AUTH_METHOD_SCORES = {
        'password': 0.3,
        'MFA': 0.7,
        'mfa': 0.7,
        'biometric': 1.0,
        'certificate': 0.9,
        'token': 0.6,
    }
    
    # Location risk mappings
    LOCATION_RISK_SCORES = {
        'low': 0.9,
        'medium': 0.5,
        'high': 0.1,
    }
    
    def __init__(self, weights: dict = None, threshold: float = 0.65):
        """
        Initialize trust scorer.
        
        Args:
            weights: Custom weight dictionary. Uses defaults if None.
            threshold: Trust threshold for ALLOW/DENY.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.threshold = threshold
        
        # Validate weights sum to ~1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
            for k in self.weights:
                self.weights[k] /= weight_sum
    
    def compute_device_score(self, device_trust: float, patch_days: int,
                              vpn_active: bool) -> float:
        """
        Compute device health score.
        
        Args:
            device_trust: Base device trust score (0-1).
            patch_days: Days since last system patch.
            vpn_active: Whether VPN is active.
            
        Returns:
            Device score between 0 and 1.
        """
        patch_penalty = min(patch_days / 365.0, 1.0)
        patch_score = 1.0 - patch_penalty
        
        vpn_bonus = 0.1 if vpn_active else 0.0
        
        score = 0.5 * device_trust + 0.3 * patch_score + 0.2 * vpn_bonus + vpn_bonus * 0.5
        return float(np.clip(score, 0.0, 1.0))
    
    def compute_user_behavior_score(self, failed_attempts: int,
                                     anomaly_score: float) -> float:
        """
        Compute user behavior score based on recent activity.
        
        Args:
            failed_attempts: Number of failed login attempts in last hour.
            anomaly_score: Anomaly detection score (0=normal, 1=anomalous).
            
        Returns:
            User behavior score between 0 and 1.
        """
        # Penalize failed attempts exponentially
        attempt_score = max(0, 1.0 - (failed_attempts * 0.15))
        
        # Invert anomaly score (lower anomaly = higher trust)
        normal_score = 1.0 - anomaly_score
        
        score = 0.6 * attempt_score + 0.4 * normal_score
        return float(np.clip(score, 0.0, 1.0))
    
    def compute_network_score(self, vpn_active: bool, source_segment: str = 'internal') -> float:
        """
        Compute network risk score.
        
        Args:
            vpn_active: Whether VPN is active.
            source_segment: Network segment of the request.
            
        Returns:
            Network score between 0 and 1.
        """
        segment_scores = {
            'internal': 0.9,
            'corporate': 0.8,
            'dmz': 0.5,
            'public': 0.2,
            'external': 0.1,
        }
        
        segment_score = segment_scores.get(source_segment.lower(), 0.3)
        vpn_bonus = 0.2 if vpn_active else 0.0
        
        return float(np.clip(segment_score + vpn_bonus, 0.0, 1.0))
    
    def compute_auth_method_score(self, auth_method: str) -> float:
        """
        Score authentication method strength.
        
        Args:
            auth_method: Authentication method used.
            
        Returns:
            Auth method score between 0 and 1.
        """
        return self.AUTH_METHOD_SCORES.get(auth_method.lower(), 0.3)
    
    def compute_location_score(self, location_risk: str) -> float:
        """
        Score based on access location risk level.
        
        Args:
            location_risk: Risk level (low/medium/high).
            
        Returns:
            Location score between 0 and 1.
        """
        return self.LOCATION_RISK_SCORES.get(location_risk.lower(), 0.5)
    
    def compute_trust_score(self, context: dict) -> dict:
        """
        Compute the complete trust score from a context dictionary.
        
        Expected context keys:
        - device_trust_score (float)
        - patch_level (int, days)
        - vpn_active (bool)
        - failed_attempts (int)
        - anomaly_score (float)
        - auth_method (str)
        - location_risk (str)
        
        Args:
            context: Dictionary with authentication context.
            
        Returns:
            Trust score result with breakdown.
        """
        # Compute component scores
        device_score = self.compute_device_score(
            context.get('device_trust_score', 0.5),
            context.get('patch_level', 30),
            context.get('vpn_active', False)
        )
        
        behavior_score = self.compute_user_behavior_score(
            context.get('failed_attempts', 0),
            context.get('anomaly_score', 0.3)
        )
        
        network_score = self.compute_network_score(
            context.get('vpn_active', False),
            context.get('source_segment', 'internal')
        )
        
        auth_score = self.compute_auth_method_score(
            context.get('auth_method', 'password')
        )
        
        location_score = self.compute_location_score(
            context.get('location_risk', 'medium')
        )
        
        # Weighted combination
        components = {
            'device_score': device_score,
            'user_behavior_score': behavior_score,
            'network_risk_score': network_score,
            'auth_method_score': auth_score,
            'location_score': location_score,
        }
        
        trust_score = sum(
            self.weights[k] * v for k, v in components.items()
        )
        trust_score = float(np.clip(trust_score, 0.0, 1.0))
        
        decision = 'ALLOW' if trust_score >= self.threshold else 'DENY'
        
        result = {
            'trust_score': round(trust_score, 4),
            'decision': decision,
            'threshold': self.threshold,
            'margin': round(trust_score - self.threshold, 4),
            'components': {
                k: {'score': round(v, 4), 'weight': self.weights[k],
                     'contribution': round(self.weights[k] * v, 4)}
                for k, v in components.items()
            },
        }
        
        return result
    
    def get_remediation_suggestions(self, trust_result: dict) -> list:
        """
        Suggest actions to improve trust score for denied requests.
        
        Args:
            trust_result: Trust score result from compute_trust_score().
            
        Returns:
            List of remediation suggestions with expected score improvement.
        """
        if trust_result['decision'] == 'ALLOW':
            return []
        
        suggestions = []
        gap = self.threshold - trust_result['trust_score']
        
        components = trust_result['components']
        
        # Sort by potential improvement (weight * (1 - current_score))
        improvement_potential = []
        for name, comp in components.items():
            potential = comp['weight'] * (1.0 - comp['score'])
            improvement_potential.append((name, potential, comp['score']))
        
        improvement_potential.sort(key=lambda x: x[1], reverse=True)
        
        remediation_map = {
            'device_score': 'Update device patches and enable endpoint protection',
            'user_behavior_score': 'Reset failed attempts counter and verify identity',
            'network_risk_score': 'Connect via VPN from a trusted network segment',
            'auth_method_score': 'Enable MFA or biometric authentication',
            'location_score': 'Access from a low-risk location or use VPN',
        }
        
        for name, potential, current in improvement_potential:
            if potential > 0.01:
                suggestions.append({
                    'action': remediation_map.get(name, f'Improve {name}'),
                    'component': name,
                    'current_score': round(current, 4),
                    'potential_improvement': round(potential, 4),
                })
        
        return suggestions
