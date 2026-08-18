"""
Feature engineering for Zero Trust Authentication.

Builds ZTA-specific features from raw authentication event data,
including trust indicators, behavioral patterns, and risk signals.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ZTAFeatureEngineer:
    """Build Zero Trust Architecture-specific features for authentication."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.engineered_feature_names = []
    
    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            df: DataFrame with base authentication features.
            
        Returns:
            DataFrame with all engineered features added.
        """
        df = df.copy()
        df = self.add_traffic_features(df)
        df = self.add_temporal_features(df)
        df = self.add_risk_indicators(df)
        df = self.add_trust_features(df)
        
        logger.info(f"Engineered {len(self.engineered_feature_names)} new features")
        return df
    
    def add_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add network traffic-based features.
        
        Features:
        - bytes_ratio: ratio of source to destination bytes
        - packet_ratio: ratio of source to destination packets
        - bytes_per_packet_src: average bytes per source packet
        - bytes_per_packet_dst: average bytes per destination packet
        """
        if 'sbytes' in df.columns and 'dbytes' in df.columns:
            df['bytes_ratio'] = df['sbytes'] / (df['dbytes'] + 1e-10)
            self.engineered_feature_names.append('bytes_ratio')
        
        if 'spkts' in df.columns and 'dpkts' in df.columns:
            df['packet_ratio'] = df['spkts'] / (df['dpkts'] + 1e-10)
            self.engineered_feature_names.append('packet_ratio')
        
        if 'sbytes' in df.columns and 'spkts' in df.columns:
            df['bytes_per_packet_src'] = df['sbytes'] / (df['spkts'] + 1e-10)
            self.engineered_feature_names.append('bytes_per_packet_src')
        
        if 'dbytes' in df.columns and 'dpkts' in df.columns:
            df['bytes_per_packet_dst'] = df['dbytes'] / (df['dpkts'] + 1e-10)
            self.engineered_feature_names.append('bytes_per_packet_dst')
        
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features for behavioral analysis.
        
        Features:
        - time_of_access_sin/cos: cyclical encoding of access hour
        - is_business_hours: boolean flag for 9am-5pm access
        - is_weekend: boolean flag (simulated from access patterns)
        """
        if 'time_of_access' in df.columns:
            hours = df['time_of_access']
            df['time_sin'] = np.sin(2 * np.pi * hours / 24)
            df['time_cos'] = np.cos(2 * np.pi * hours / 24)
            df['is_business_hours'] = ((hours >= 9) & (hours <= 17)).astype(int)
            self.engineered_feature_names.extend(
                ['time_sin', 'time_cos', 'is_business_hours']
            )
        
        return df
    
    def add_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add risk indicator features for ZTA decision making.
        
        Features:
        - high_risk_flag: composite risk indicator
        - failed_attempt_severity: categorized failed attempt levels
        - patch_risk: risk level based on days since last patch
        """
        if 'failed_attempts' in df.columns:
            df['failed_attempt_severity'] = pd.cut(
                df['failed_attempts'],
                bins=[-1, 0, 2, 5, float('inf')],
                labels=[0, 1, 2, 3]
            ).astype(int)
            self.engineered_feature_names.append('failed_attempt_severity')
        
        if 'patch_level' in df.columns:
            df['patch_risk'] = pd.cut(
                df['patch_level'],
                bins=[-1, 7, 30, 90, float('inf')],
                labels=[0, 1, 2, 3]
            ).astype(int)
            self.engineered_feature_names.append('patch_risk')
        
        # Composite high-risk flag
        risk_conditions = []
        if 'failed_attempts' in df.columns:
            risk_conditions.append(df['failed_attempts'] > 3)
        if 'device_trust_score' in df.columns:
            risk_conditions.append(df['device_trust_score'] < 0.4)
        if 'anomaly_score' in df.columns:
            risk_conditions.append(df['anomaly_score'] > 0.7)
        
        if risk_conditions:
            df['high_risk_flag'] = np.where(
                np.logical_or.reduce(risk_conditions), 1, 0
            )
            self.engineered_feature_names.append('high_risk_flag')
        
        return df
    
    def add_trust_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trust-related composite features.
        
        Features:
        - auth_strength: numeric encoding of authentication method strength
        - device_health_score: composite device health metric
        - access_risk_score: combined risk from location and resource sensitivity
        """
        if 'auth_method' in df.columns:
            auth_strength_map = {'password': 0.3, 'MFA': 0.7, 'biometric': 1.0}
            if not pd.api.types.is_numeric_dtype(df['auth_method']):
                df['auth_strength'] = df['auth_method'].map(auth_strength_map).fillna(0.5)
            else:
                max_val = df['auth_method'].max()
                df['auth_strength'] = df['auth_method'] / max_val if max_val > 0 else 0.5
            self.engineered_feature_names.append('auth_strength')
        
        if 'device_trust_score' in df.columns and 'patch_level' in df.columns:
            # Higher patch_level (more days since patch) = lower health
            max_patch = df['patch_level'].max() if df['patch_level'].max() > 0 else 1
            patch_score = 1.0 - (df['patch_level'] / max_patch)
            df['device_health_score'] = (df['device_trust_score'] + patch_score) / 2
            self.engineered_feature_names.append('device_health_score')
        
        if 'location_risk' in df.columns and 'resource_sensitivity' in df.columns:
            loc_risk_map = {'low': 0.2, 'medium': 0.5, 'high': 0.9}
            if not pd.api.types.is_numeric_dtype(df['location_risk']):
                loc_numeric = df['location_risk'].map(loc_risk_map).fillna(0.5)
            else:
                loc_numeric = df['location_risk']
            max_sensitivity = df['resource_sensitivity'].max() if df['resource_sensitivity'].max() > 0 else 1
            df['access_risk_score'] = (
                loc_numeric * 0.5 + (df['resource_sensitivity'] / max_sensitivity) * 0.5
            )
            self.engineered_feature_names.append('access_risk_score')
        
        return df
    
    def get_feature_descriptions(self) -> dict:
        """
        Return human-readable descriptions of all engineered features.
        
        Returns:
            Dictionary mapping feature names to descriptions.
        """
        return {
            'bytes_ratio': 'Ratio of source to destination bytes transferred',
            'packet_ratio': 'Ratio of source to destination packet counts',
            'bytes_per_packet_src': 'Average bytes per source packet',
            'bytes_per_packet_dst': 'Average bytes per destination packet',
            'time_sin': 'Sine component of cyclical time encoding',
            'time_cos': 'Cosine component of cyclical time encoding',
            'is_business_hours': 'Whether access occurred during business hours (9am-5pm)',
            'failed_attempt_severity': 'Categorized severity of recent failed attempts',
            'patch_risk': 'Risk level based on days since last system patch',
            'high_risk_flag': 'Composite flag indicating high-risk access attempt',
            'auth_strength': 'Numeric encoding of authentication method strength',
            'device_health_score': 'Composite device health from trust and patch level',
            'access_risk_score': 'Combined risk from location and resource sensitivity',
        }
