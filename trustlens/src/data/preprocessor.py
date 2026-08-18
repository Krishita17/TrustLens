"""
Data preprocessor for TrustLens authentication events.

Handles cleaning, encoding categorical variables, normalizing numeric features,
and handling class imbalance for the authentication decision dataset.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class AuthDataPreprocessor:
    """Preprocessor for authentication event data."""
    
    def __init__(self, scaling_method: str = 'standard', random_state: int = 42):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: 'standard' for StandardScaler or 'minmax' for MinMaxScaler.
            random_state: Random seed for reproducibility.
        """
        self.scaling_method = scaling_method
        self.random_state = random_state
        self.scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        self.label_encoders = {}
        self.feature_names = []
        self._is_fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw authentication data.
        
        Steps:
        1. Remove duplicate rows
        2. Handle missing values
        3. Remove obviously invalid records
        
        Args:
            df: Raw DataFrame.
            
        Returns:
            Cleaned DataFrame.
        """
        initial_len = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_len - len(df)} duplicate rows")
        
        # Handle missing values: fill numeric with median, categorical with mode
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'unknown')
        
        # Remove rows with negative duration or packet counts (if columns exist)
        for col in ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes']:
            if col in df.columns:
                df = df[df[col] >= 0]
        
        logger.info(f"Cleaned data: {len(df)} records remaining from {initial_len}")
        return df.reset_index(drop=True)
    
    def encode_categorical(self, df: pd.DataFrame, categorical_cols: list = None) -> pd.DataFrame:
        """
        Encode categorical variables using LabelEncoder.
        
        Args:
            df: DataFrame with categorical columns.
            categorical_cols: List of columns to encode. Auto-detected if None.
            
        Returns:
            DataFrame with encoded categorical variables.
        """
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        df = df.copy()
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories during transform
                    le = self.label_encoders[col]
                    known = set(le.classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df[col] = le.transform(df[col].astype(str))
                    
                logger.info(f"Encoded '{col}': {len(self.label_encoders[col].classes_)} categories")
        
        return df
    
    def scale_features(self, df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
        """
        Scale numeric features using the configured scaler.
        
        Args:
            df: DataFrame with numeric features.
            exclude_cols: Columns to exclude from scaling (e.g., labels).
            
        Returns:
            DataFrame with scaled features.
        """
        if exclude_cols is None:
            exclude_cols = []
        
        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]
        
        df = df.copy()
        if not self._is_fitted:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            self._is_fitted = True
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        self.feature_names = numeric_cols
        logger.info(f"Scaled {len(numeric_cols)} numeric features using {self.scaling_method}")
        return df
    
    def prepare_train_test(self, df: pd.DataFrame, target_col: str = 'decision',
                           test_size: float = 0.2) -> tuple:
        """
        Split data into train and test sets.
        
        Args:
            df: Preprocessed DataFrame.
            target_col: Name of the target column.
            test_size: Fraction for test set.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(
            f"Train/test split: {len(X_train)} train, {len(X_test)} test "
            f"(positive rate: {y_train.mean():.3f} / {y_test.mean():.3f})"
        )
        return X_train, X_test, y_train, y_test
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'decision') -> pd.DataFrame:
        """
        Run full preprocessing pipeline: clean → encode → scale.
        
        Args:
            df: Raw DataFrame.
            target_col: Target column name (excluded from scaling).
            
        Returns:
            Fully preprocessed DataFrame.
        """
        df = self.clean_data(df)
        df = self.encode_categorical(df)
        df = self.scale_features(df, exclude_cols=[target_col])
        return df
