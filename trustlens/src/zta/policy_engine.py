"""
Zero Trust Architecture Policy Engine for TrustLens.

Implements NIST SP 800-207 Zero Trust principles:
- Never trust, always verify
- Least privilege access
- Continuous validation
- Micro-segmentation
- Trust score computation
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class ZTAPolicyEngine:
    """Zero Trust Architecture policy enforcement engine."""
    
    # Role-based access levels
    ROLE_CLEARANCE = {
        'viewer': 1,
        'user': 2,
        'analyst': 3,
        'admin': 4,
        'superadmin': 5,
    }
    
    # Network segments
    NETWORK_SEGMENTS = {
        'public': ['public', 'dmz'],
        'internal': ['internal', 'corporate'],
        'restricted': ['restricted', 'secure'],
        'critical': ['critical', 'scada'],
    }
    
    def __init__(self, trust_threshold: float = 0.65, reauth_interval_minutes: int = 15):
        """
        Initialize the ZTA policy engine.
        
        Args:
            trust_threshold: Minimum trust score for ALLOW decisions.
            reauth_interval_minutes: Minutes before re-authentication required.
        """
        self.trust_threshold = trust_threshold
        self.reauth_interval = timedelta(minutes=reauth_interval_minutes)
        self.active_sessions = {}  # session_id -> last_auth_time
        self.decision_log = []
        
        logger.info(
            f"ZTA Policy Engine initialized: threshold={trust_threshold}, "
            f"reauth_interval={reauth_interval_minutes}min"
        )
    
    def never_trust_always_verify(self, request: dict) -> dict:
        """
        Core ZTA principle: Every request is evaluated regardless of previous access.
        
        No implicit trust based on network location, previous authentication,
        or any other factor.
        
        Args:
            request: Authentication request with user and context info.
            
        Returns:
            Verification result dict with status and reasons.
        """
        required_fields = ['user_id', 'resource_id', 'timestamp']
        missing = [f for f in required_fields if f not in request]
        
        if missing:
            return {
                'verified': False,
                'reason': f'Missing required fields: {missing}',
                'policy': 'never_trust_always_verify'
            }
        
        return {
            'verified': True,
            'reason': 'Request contains all required fields for verification',
            'policy': 'never_trust_always_verify'
        }
    
    def least_privilege_check(self, user_role: str, resource_sensitivity: int) -> dict:
        """
        Enforce least privilege — deny if user requests resource above clearance.
        
        Args:
            user_role: User's role (viewer, user, analyst, admin, superadmin).
            resource_sensitivity: Resource sensitivity level (1-5).
            
        Returns:
            Policy check result dict.
        """
        clearance = self.ROLE_CLEARANCE.get(user_role.lower(), 0)
        
        if clearance >= resource_sensitivity:
            return {
                'allowed': True,
                'reason': f'User role "{user_role}" (level {clearance}) has sufficient '
                          f'clearance for resource (sensitivity {resource_sensitivity})',
                'policy': 'least_privilege'
            }
        else:
            return {
                'allowed': False,
                'reason': f'User role "{user_role}" (level {clearance}) lacks clearance '
                          f'for resource (sensitivity {resource_sensitivity})',
                'policy': 'least_privilege'
            }
    
    def continuous_validation(self, session_id: str, current_time: datetime = None) -> dict:
        """
        Check if session requires re-authentication.
        
        Args:
            session_id: Active session identifier.
            current_time: Current timestamp (default: now).
            
        Returns:
            Validation result dict.
        """
        if current_time is None:
            current_time = datetime.now()
        
        if session_id not in self.active_sessions:
            return {
                'valid': False,
                'reason': 'No active session found — initial authentication required',
                'requires_reauth': True,
                'policy': 'continuous_validation'
            }
        
        last_auth = self.active_sessions[session_id]
        time_since_auth = current_time - last_auth
        
        if time_since_auth > self.reauth_interval:
            return {
                'valid': False,
                'reason': f'Session expired — last auth was {time_since_auth.seconds // 60} minutes ago',
                'requires_reauth': True,
                'policy': 'continuous_validation'
            }
        
        return {
            'valid': True,
            'reason': f'Session valid — {(self.reauth_interval - time_since_auth).seconds // 60} minutes remaining',
            'requires_reauth': False,
            'policy': 'continuous_validation'
        }
    
    def refresh_session(self, session_id: str, auth_time: datetime = None):
        """
        Update session's last authentication time.
        
        Args:
            session_id: Session to refresh.
            auth_time: Authentication timestamp (default: now).
        """
        if auth_time is None:
            auth_time = datetime.now()
        self.active_sessions[session_id] = auth_time
    
    def micro_segmentation_check(self, source_segment: str, target_segment: str) -> dict:
        """
        Enforce network micro-segmentation boundaries.
        
        Args:
            source_segment: Source network segment.
            target_segment: Target network segment.
            
        Returns:
            Segmentation check result dict.
        """
        # Define allowed transitions
        allowed_transitions = {
            'public': ['public', 'dmz'],
            'dmz': ['public', 'dmz', 'internal'],
            'internal': ['internal', 'dmz', 'corporate'],
            'corporate': ['corporate', 'internal'],
            'restricted': ['restricted', 'internal'],
            'secure': ['secure', 'restricted'],
            'critical': ['critical'],
            'scada': ['scada'],
        }
        
        allowed = allowed_transitions.get(source_segment.lower(), [])
        
        if target_segment.lower() in allowed:
            return {
                'allowed': True,
                'reason': f'Transition {source_segment} → {target_segment} is permitted',
                'policy': 'micro_segmentation'
            }
        else:
            return {
                'allowed': False,
                'reason': f'Transition {source_segment} → {target_segment} violates segmentation policy',
                'policy': 'micro_segmentation'
            }
    
    def compute_trust_score(self, context: dict) -> dict:
        """
        Compute weighted trust score from context vector.
        
        Trust = 0.30 * device_score +
                0.25 * user_behavior_score +
                0.20 * network_risk_score +
                0.15 * auth_method_score +
                0.10 * location_score
        
        Args:
            context: Dictionary with score components (0.0 to 1.0 each).
            
        Returns:
            Trust score result with decision.
        """
        weights = {
            'device_score': 0.30,
            'user_behavior_score': 0.25,
            'network_risk_score': 0.20,
            'auth_method_score': 0.15,
            'location_score': 0.10,
        }
        
        trust_score = 0.0
        components = {}
        
        for component, weight in weights.items():
            value = context.get(component, 0.5)  # Default to 0.5 if missing
            value = max(0.0, min(1.0, value))  # Clamp to [0, 1]
            trust_score += weight * value
            components[component] = {'value': round(value, 4), 'weight': weight, 'contribution': round(weight * value, 4)}
        
        decision = 'ALLOW' if trust_score >= self.trust_threshold else 'DENY'
        
        result = {
            'trust_score': round(trust_score, 4),
            'threshold': self.trust_threshold,
            'decision': decision,
            'components': components,
            'margin': round(trust_score - self.trust_threshold, 4),
        }
        
        logger.info(f"Trust score: {trust_score:.4f} → {decision} (threshold: {self.trust_threshold})")
        return result
    
    def evaluate_request(self, request: dict) -> dict:
        """
        Full ZTA policy evaluation for an authentication request.
        
        Runs all policy checks and returns comprehensive result.
        
        Args:
            request: Full authentication request with context.
            
        Returns:
            Complete evaluation result with all policy checks.
        """
        results = {
            'request_id': request.get('request_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'overall_decision': 'ALLOW',
            'denial_reasons': [],
        }
        
        # 1. Never trust, always verify
        verify = self.never_trust_always_verify(request)
        results['checks']['verify'] = verify
        if not verify['verified']:
            results['overall_decision'] = 'DENY'
            results['denial_reasons'].append(verify['reason'])
        
        # 2. Least privilege
        if 'user_role' in request and 'resource_sensitivity' in request:
            priv = self.least_privilege_check(request['user_role'], request['resource_sensitivity'])
            results['checks']['least_privilege'] = priv
            if not priv['allowed']:
                results['overall_decision'] = 'DENY'
                results['denial_reasons'].append(priv['reason'])
        
        # 3. Continuous validation
        if 'session_id' in request:
            session = self.continuous_validation(request['session_id'])
            results['checks']['continuous_validation'] = session
            if not session['valid']:
                results['overall_decision'] = 'DENY'
                results['denial_reasons'].append(session['reason'])
        
        # 4. Micro-segmentation
        if 'source_segment' in request and 'target_segment' in request:
            seg = self.micro_segmentation_check(request['source_segment'], request['target_segment'])
            results['checks']['micro_segmentation'] = seg
            if not seg['allowed']:
                results['overall_decision'] = 'DENY'
                results['denial_reasons'].append(seg['reason'])
        
        # 5. Trust score
        if 'context' in request:
            trust = self.compute_trust_score(request['context'])
            results['checks']['trust_score'] = trust
            results['trust_score'] = trust['trust_score']
            if trust['decision'] == 'DENY':
                results['overall_decision'] = 'DENY'
                results['denial_reasons'].append(
                    f"Trust score {trust['trust_score']:.4f} below threshold {self.trust_threshold}"
                )
        
        return results
