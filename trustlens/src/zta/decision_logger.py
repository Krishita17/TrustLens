"""
Decision Logger for TrustLens with compliance mapping.

Logs every authentication decision with timestamp, trust score,
explanation, and compliance framework mapping (NIST, HIPAA, GDPR).
"""

import os
import csv
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class DecisionLogger:
    """Log and audit authentication decisions with compliance mapping."""
    
    # NIST SP 800-207 ZTA Pillars
    NIST_PILLARS = {
        'identity': 'User identity and authentication verification',
        'device': 'Device health and compliance status',
        'network': 'Network segmentation and access controls',
        'application': 'Application-level security policies',
        'data': 'Data classification and protection level',
    }
    
    def __init__(self, log_dir: str = None):
        """
        Initialize decision logger.
        
        Args:
            log_dir: Directory for log files. Uses default if None.
        """
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'experiments', 'logs'
            )
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, 'decision_audit.csv')
        self.decisions = []
        
        # Initialize CSV with headers if needed
        if not os.path.exists(self.log_file):
            self._write_csv_header()
        
        logger.info(f"Decision logger initialized, logging to {self.log_file}")
    
    def _write_csv_header(self):
        """Write CSV header row."""
        headers = [
            'request_id', 'timestamp', 'user_id', 'decision', 'trust_score',
            'denial_reasons', 'nist_pillar', 'hipaa_flag', 'gdpr_basis',
            'explanation_summary', 'compliance_notes'
        ]
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_decision(self, request_id: str, decision: str, trust_score: float,
                     explanation: dict = None, context: dict = None) -> dict:
        """
        Log an authentication decision with full audit trail.
        
        Args:
            request_id: Unique request identifier.
            decision: ALLOW or DENY.
            trust_score: Computed trust score.
            explanation: XAI explanation dict.
            context: Authentication context.
            
        Returns:
            Complete log entry dict.
        """
        timestamp = datetime.now().isoformat()
        
        # Map to NIST pillars
        nist_pillar = self._map_nist_pillar(context or {})
        
        # Check HIPAA applicability
        hipaa_flag = self._check_hipaa(context or {})
        
        # GDPR basis
        gdpr_basis = self._determine_gdpr_basis(decision, context or {})
        
        # Build explanation summary
        explanation_summary = self._summarize_explanation(explanation or {})
        
        entry = {
            'request_id': request_id,
            'timestamp': timestamp,
            'user_id': context.get('user_id', 'unknown') if context else 'unknown',
            'decision': decision,
            'trust_score': round(trust_score, 4),
            'denial_reasons': json.dumps(context.get('denial_reasons', [])) if context else '[]',
            'nist_pillar': nist_pillar,
            'hipaa_flag': hipaa_flag,
            'gdpr_basis': gdpr_basis,
            'explanation_summary': explanation_summary,
            'compliance_notes': self._build_compliance_notes(decision, context or {}),
        }
        
        # Append to CSV
        self._append_to_csv(entry)
        
        # Keep in memory
        self.decisions.append(entry)
        
        logger.info(f"Logged decision: {request_id} → {decision} (trust: {trust_score:.4f})")
        return entry
    
    def _append_to_csv(self, entry: dict):
        """Append a log entry to the CSV file."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([entry.get(k, '') for k in [
                'request_id', 'timestamp', 'user_id', 'decision', 'trust_score',
                'denial_reasons', 'nist_pillar', 'hipaa_flag', 'gdpr_basis',
                'explanation_summary', 'compliance_notes'
            ]])
    
    def _map_nist_pillar(self, context: dict) -> str:
        """
        Map decision to NIST SP 800-207 ZTA pillar.
        
        Args:
            context: Authentication context.
            
        Returns:
            Primary NIST pillar that triggered the decision.
        """
        if context.get('failed_attempts', 0) > 3:
            return 'identity'
        elif context.get('device_trust_score', 1.0) < 0.5:
            return 'device'
        elif context.get('source_segment') in ['public', 'external']:
            return 'network'
        elif context.get('resource_sensitivity', 0) >= 4:
            return 'data'
        else:
            return 'application'
    
    def _check_hipaa(self, context: dict) -> bool:
        """
        Check if the decision involves HIPAA-relevant resources.
        
        Args:
            context: Authentication context.
            
        Returns:
            True if HIPAA-relevant (PHI-adjacent resource).
        """
        # Flag high-sensitivity resources as potentially PHI-adjacent
        return context.get('resource_sensitivity', 0) >= 4
    
    def _determine_gdpr_basis(self, decision: str, context: dict) -> str:
        """
        Determine GDPR legal basis for the access decision.
        
        Args:
            decision: ALLOW or DENY.
            context: Authentication context.
            
        Returns:
            GDPR legal basis string.
        """
        if decision == 'ALLOW':
            return 'Legitimate interest — authorized access to required resource'
        else:
            reasons = []
            if context.get('trust_score', 1.0) < 0.65:
                reasons.append('Insufficient trust score for data access')
            if context.get('failed_attempts', 0) > 3:
                reasons.append('Suspicious authentication activity detected')
            return '; '.join(reasons) if reasons else 'Access denied per security policy'
    
    def _summarize_explanation(self, explanation: dict) -> str:
        """
        Create a concise summary of the XAI explanation.
        
        Args:
            explanation: XAI explanation dictionary.
            
        Returns:
            Summary string.
        """
        if not explanation:
            return 'No explanation available'
        
        if 'top_features' in explanation:
            top = explanation['top_features'][:3]
            parts = [f"{t['feature']}={t.get('feature_value', 'N/A')}" for t in top]
            return f"Top factors: {', '.join(parts)}"
        elif 'text_summary' in explanation:
            return explanation['text_summary'][:200]
        elif 'anchor_rules' in explanation:
            return ' AND '.join(explanation['anchor_rules'][:3])
        
        return 'Explanation generated'
    
    def _build_compliance_notes(self, decision: str, context: dict) -> str:
        """Build compliance notes for the log entry."""
        notes = []
        
        if context.get('resource_sensitivity', 0) >= 4:
            notes.append('High-sensitivity resource — minimum necessary access enforced')
        
        if decision == 'DENY':
            notes.append('Data subject right to explanation: denial reason provided')
        
        return '; '.join(notes) if notes else 'Standard access evaluation'
    
    def get_audit_log(self) -> list:
        """Return all logged decisions."""
        return self.decisions
    
    def export_audit_csv(self, output_path: str = None) -> str:
        """
        Export audit log to a new CSV file.
        
        Args:
            output_path: Output file path. Uses timestamped default if None.
            
        Returns:
            Path to exported file.
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.log_dir, f'audit_export_{timestamp}.csv')
        
        import pandas as pd
        if self.decisions:
            pd.DataFrame(self.decisions).to_csv(output_path, index=False)
            logger.info(f"Exported {len(self.decisions)} decisions to {output_path}")
        
        return output_path
