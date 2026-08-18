"""
Synthetic authentication event generator for TrustLens research.

Generates realistic Zero Trust authentication events with configurable
distributions and correlations between features. The generated data
maintains realistic relationships (e.g., high failed_attempts correlates
with DENY decisions).
"""

import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Default output path
DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'synthetic', 'generated_auth_logs.csv'
)


def generate_synthetic_auth_data(n_samples: int = 50000, random_state: int = 42,
                                  output_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic ZTA authentication event data.
    
    Creates realistic authentication events with correlated features
    that reflect real-world ZTA access patterns.
    
    Args:
        n_samples: Number of events to generate.
        random_state: Random seed for reproducibility.
        output_path: Path to save CSV. If None, uses default path.
        
    Returns:
        DataFrame with generated authentication events.
    """
    np.random.seed(random_state)
    
    logger.info(f"Generating {n_samples} synthetic authentication events...")
    
    # Generate base features
    data = {
        'user_id': [f'USER_{i:05d}' for i in np.random.randint(1, 501, n_samples)],
        'device_trust_score': np.clip(np.random.beta(5, 2, n_samples), 0, 1),
        'location_risk': np.random.choice(
            ['low', 'medium', 'high'], n_samples, p=[0.5, 0.35, 0.15]
        ),
        'time_of_access': np.random.randint(0, 24, n_samples),
        'failed_attempts': np.random.poisson(1.5, n_samples),
        'resource_sensitivity': np.random.randint(1, 6, n_samples),
        'vpn_active': np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
        'patch_level': np.random.exponential(15, n_samples).astype(int),
        'auth_method': np.random.choice(
            ['password', 'MFA', 'biometric'], n_samples, p=[0.3, 0.5, 0.2]
        ),
    }
    
    df = pd.DataFrame(data)
    
    # Cap values
    df['failed_attempts'] = df['failed_attempts'].clip(0, 20)
    df['patch_level'] = df['patch_level'].clip(0, 365)
    
    # Generate anomaly score (correlated with risk factors)
    location_risk_numeric = df['location_risk'].map(
        {'low': 0.1, 'medium': 0.4, 'high': 0.8}
    )
    base_anomaly = (
        0.3 * (1 - df['device_trust_score']) +
        0.25 * location_risk_numeric +
        0.2 * (df['failed_attempts'] / 20) +
        0.15 * (df['patch_level'] / 365) +
        0.1 * np.random.uniform(0, 1, n_samples)
    )
    df['anomaly_score'] = np.clip(base_anomaly, 0, 1).round(4)
    
    # Generate decision based on realistic rules
    auth_strength = df['auth_method'].map(
        {'password': 0.3, 'MFA': 0.7, 'biometric': 1.0}
    )
    
    trust_score = (
        0.30 * df['device_trust_score'] +
        0.25 * (1 - location_risk_numeric) +
        0.20 * (1 - df['failed_attempts'] / 20) +
        0.15 * auth_strength +
        0.10 * (1 - df['patch_level'] / 365)
    )
    
    # Add noise and threshold
    trust_score_noisy = trust_score + np.random.normal(0, 0.05, n_samples)
    df['decision'] = np.where(trust_score_noisy >= 0.65, 'ALLOW', 'DENY')
    
    # Encode decision as binary for ML (0=ALLOW, 1=DENY)
    df['decision_binary'] = (df['decision'] == 'DENY').astype(int)
    
    # Log distribution
    allow_count = (df['decision'] == 'ALLOW').sum()
    deny_count = (df['decision'] == 'DENY').sum()
    logger.info(f"Generated: {allow_count} ALLOW ({allow_count/n_samples*100:.1f}%), "
                f"{deny_count} DENY ({deny_count/n_samples*100:.1f}%)")
    
    # Save to CSV
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved synthetic data to {output_path}")
    
    return df


def generate_unsw_nb15_adapted(n_samples: int = 50000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data that mimics the UNSW-NB15 dataset structure
    adapted for ZTA authentication context.
    
    Args:
        n_samples: Number of records to generate.
        random_state: Random seed.
        
    Returns:
        DataFrame with UNSW-NB15 style features.
    """
    np.random.seed(random_state)
    
    data = {
        'dur': np.abs(np.random.exponential(2.0, n_samples)).round(4),
        'proto': np.random.choice(['tcp', 'udp', 'icmp', 'arp'], n_samples, p=[0.6, 0.25, 0.1, 0.05]),
        'service': np.random.choice(['-', 'http', 'dns', 'smtp', 'ssh', 'ftp', 'ssl'], n_samples, p=[0.3, 0.25, 0.15, 0.08, 0.1, 0.05, 0.07]),
        'state': np.random.choice(['FIN', 'CON', 'INT', 'RST', 'REQ'], n_samples, p=[0.35, 0.25, 0.15, 0.15, 0.1]),
        'spkts': np.random.poisson(10, n_samples),
        'dpkts': np.random.poisson(8, n_samples),
        'sbytes': np.random.exponential(500, n_samples).astype(int),
        'dbytes': np.random.exponential(800, n_samples).astype(int),
        'sttl': np.random.choice([62, 64, 128, 254, 255], n_samples, p=[0.15, 0.35, 0.3, 0.1, 0.1]),
        'dttl': np.random.choice([62, 64, 128, 252, 254], n_samples, p=[0.1, 0.3, 0.35, 0.15, 0.1]),
        'sloss': np.random.poisson(1, n_samples),
        'dloss': np.random.poisson(1, n_samples),
        'sinpkt': np.abs(np.random.exponential(50, n_samples)).round(2),
        'dinpkt': np.abs(np.random.exponential(50, n_samples)).round(2),
        'ct_srv_src': np.random.poisson(5, n_samples),
        'ct_dst_ltm': np.random.poisson(3, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate labels: 0=normal(ALLOW), 1=attack(DENY)
    attack_prob = 0.3 + 0.2 * (df['sloss'] > 2).astype(float) + 0.1 * (df['dur'] > 5).astype(float)
    attack_prob = np.clip(attack_prob + np.random.normal(0, 0.1, n_samples), 0.05, 0.95)
    df['label'] = (np.random.uniform(0, 1, n_samples) < attack_prob).astype(int)
    
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Generate ZTA synthetic data
    print("Generating ZTA synthetic authentication data...")
    zta_data = generate_synthetic_auth_data(n_samples=50000)
    print(f"ZTA data shape: {zta_data.shape}")
    print(f"Decision distribution:\n{zta_data['decision'].value_counts()}")
    
    # Generate UNSW-NB15 adapted data
    print("\nGenerating UNSW-NB15 adapted data...")
    unsw_data = generate_unsw_nb15_adapted(n_samples=50000)
    print(f"UNSW data shape: {unsw_data.shape}")
    print(f"Label distribution:\n{unsw_data['label'].value_counts()}")
