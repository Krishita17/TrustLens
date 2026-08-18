"""
Context Builder for ZTA authentication requests.

Constructs the context vector used for trust score computation
from raw authentication event data.
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Build context vectors for ZTA trust scoring."""
    
    def __init__(self):
        """Initialize the context builder."""
        self.user_profiles = {}  # Cache of user behavior profiles
    
    def build_context(self, auth_event: dict) -> dict:
        """
        Build a complete context vector from an authentication event.
        
        Args:
            auth_event: Raw authentication event dictionary with fields like
                        user_id, device_trust_score, location_risk, etc.
            
        Returns:
            Context dictionary ready for trust scoring.
        """
        context = {
            'user_id': auth_event.get('user_id', 'unknown'),
            'device_trust_score': float(auth_event.get('device_trust_score', 0.5)),
            'patch_level': int(auth_event.get('patch_level', 30)),
            'vpn_active': bool(auth_event.get('vpn_active', False)),
            'failed_attempts': int(auth_event.get('failed_attempts', 0)),
            'anomaly_score': float(auth_event.get('anomaly_score', 0.3)),
            'auth_method': str(auth_event.get('auth_method', 'password')),
            'location_risk': str(auth_event.get('location_risk', 'medium')),
            'resource_sensitivity': int(auth_event.get('resource_sensitivity', 3)),
            'time_of_access': int(auth_event.get('time_of_access', 12)),
            'timestamp': auth_event.get('timestamp', datetime.now().isoformat()),
        }
        
        # Add derived features
        context['is_business_hours'] = 9 <= context['time_of_access'] <= 17
        context['is_high_sensitivity'] = context['resource_sensitivity'] >= 4
        
        logger.debug(f"Built context for user {context['user_id']}")
        return context
    
    def build_from_dataframe_row(self, row) -> dict:
        """
        Build context from a pandas DataFrame row.
        
        Args:
            row: pandas Series (single row from DataFrame).
            
        Returns:
            Context dictionary.
        """
        return self.build_context(row.to_dict())
    
    def enrich_context(self, context: dict, user_history: dict = None) -> dict:
        """
        Enrich context with additional behavioral and historical data.
        
        Args:
            context: Base context dictionary.
            user_history: Historical user behavior data.
            
        Returns:
            Enriched context dictionary.
        """
        enriched = context.copy()
        
        if user_history:
            enriched['avg_daily_requests'] = user_history.get('avg_daily_requests', 10)
            enriched['typical_access_hours'] = user_history.get('typical_hours', [9, 17])
            enriched['past_violations'] = user_history.get('violations', 0)
            
            # Flag unusual access patterns
            hour = enriched.get('time_of_access', 12)
            typical = enriched.get('typical_access_hours', [9, 17])
            enriched['unusual_time'] = hour < typical[0] or hour > typical[1]
        
        return enriched
    
    def context_to_feature_vector(self, context: dict, feature_order: list = None) -> list:
        """
        Convert context dictionary to numeric feature vector for ML model input.
        
        Args:
            context: Context dictionary.
            feature_order: Ordered list of feature names. Uses default if None.
            
        Returns:
            List of numeric feature values.
        """
        if feature_order is None:
            feature_order = [
                'device_trust_score', 'failed_attempts', 'resource_sensitivity',
                'vpn_active', 'patch_level', 'anomaly_score',
            ]
        
        location_map = {'low': 0, 'medium': 1, 'high': 2}
        auth_map = {'password': 0, 'MFA': 1, 'biometric': 2}
        
        vector = []
        for feat in feature_order:
            val = context.get(feat)
            if val is None:
                vector.append(0.0)
            elif isinstance(val, bool):
                vector.append(1.0 if val else 0.0)
            elif isinstance(val, str):
                if feat == 'location_risk':
                    vector.append(float(location_map.get(val, 1)))
                elif feat == 'auth_method':
                    vector.append(float(auth_map.get(val, 0)))
                else:
                    vector.append(0.0)
            else:
                vector.append(float(val))
        
        return vector
