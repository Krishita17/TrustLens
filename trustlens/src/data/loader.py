"""
Data loader for TrustLens authentication datasets.

Handles loading UNSW-NB15 raw data and synthetic authentication logs.
Validates schema, checks for missing values, and ensures data integrity.
"""

import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# UNSW-NB15 expected columns relevant to ZTA
UNSW_NB15_COLUMNS = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts',
    'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
    'sinpkt', 'dinpkt', 'ct_srv_src', 'ct_dst_ltm', 'label'
]

# Synthetic ZTA dataset columns
SYNTHETIC_COLUMNS = [
    'user_id', 'device_trust_score', 'location_risk', 'time_of_access',
    'failed_attempts', 'resource_sensitivity', 'vpn_active', 'patch_level',
    'auth_method', 'anomaly_score', 'decision'
]


def load_unsw_nb15(data_dir: str, file_pattern: str = "UNSW-NB15_*.csv") -> pd.DataFrame:
    """
    Load UNSW-NB15 dataset from one or more CSV files.
    
    Args:
        data_dir: Directory containing raw CSV files.
        file_pattern: Glob pattern for CSV files.
        
    Returns:
        Combined DataFrame with all records.
        
    Raises:
        FileNotFoundError: If no matching files found.
        ValueError: If required columns are missing.
    """
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    
    if not files:
        logger.warning(f"No files matching '{file_pattern}' in {data_dir}")
        raise FileNotFoundError(f"No UNSW-NB15 files found in {data_dir}")
    
    dfs = []
    for f in files:
        logger.info(f"Loading {f}")
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} records from {len(files)} files")
    
    # Validate required columns exist
    validate_columns(combined, UNSW_NB15_COLUMNS, dataset_name="UNSW-NB15")
    
    return combined


def load_synthetic_data(filepath: str) -> pd.DataFrame:
    """
    Load synthetic authentication event data.
    
    Args:
        filepath: Path to synthetic CSV file.
        
    Returns:
        DataFrame with synthetic authentication events.
        
    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Synthetic data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} synthetic records from {filepath}")
    
    validate_columns(df, SYNTHETIC_COLUMNS, dataset_name="Synthetic")
    
    return df


def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load pre-processed authentication event data ready for model training.
    
    Args:
        filepath: Path to processed CSV file.
        
    Returns:
        DataFrame with processed features and labels.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} processed records from {filepath}")
    
    return df


def validate_columns(df: pd.DataFrame, required_columns: list, dataset_name: str = "Dataset") -> bool:
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must be present.
        dataset_name: Name for error messages.
        
    Returns:
        True if validation passes.
        
    Raises:
        ValueError: If required columns are missing.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )
    logger.info(f"{dataset_name} schema validation passed")
    return True


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the dataset for quick inspection.
    
    Args:
        df: DataFrame to summarize.
        
    Returns:
        Dictionary with shape, dtypes, null counts, and basic stats.
    """
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'null_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'numeric_stats': df.describe().to_dict() if len(df) > 0 else {}
    }
