"""
Data loading with single train/test split.

CRITICAL WARNING TO ALL CONCERNED: All developers must use same random_state for fair comparison.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load data with fixed train/test split - NO PREPROCESSING."""
    
    def __init__(
        self,
        data_path: str = "main.csv",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.data_path = Path(data_path)
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names = [ 'learning_rate', 'batch_size', 'buffer_size', 'hidden_units', 'num_layers', 'time_horizon', 'beta', 'gamma', 'lambd', 'max_steps']
        self.valid_targets = ['final_10_percent_reward_mean', 'late_phase_std']
    
    def load_and_split(self, target: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data and return train/test split.
        
        Args:
            target: 'final_10_percent_reward_mean' or 'late_phase_std'
            
        Returns:
            (X_train, X_test, y_train, y_test) - raw data with no preprocessing done
            
        SAFETY WARNING:
            No preprocessing is applied. all preprocessing must be inside your sklearn Pipeline or you risk data leakage.
        """
        if target not in self.valid_targets:
            raise ValueError(f"Invalid target. Must be one of: {self.valid_targets}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        if 'error_occurred' in df.columns:
            df = df[~df['error_occurred'].fillna(False)]
        
        df = df.dropna(subset=self.feature_names + [target])
        
        X = df[self.feature_names].values
        y = df[target].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )
        
        print(f"\nData loaded for target: {target}")
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Test:  {X_test.shape[0]} samples")
        print(f"Features: {len(self.feature_names)}")
        print("safety warning : raw data returned (no preprocessing)")
        print("all preprocessing must be inside sklearn.pipeline.Pipeline")
        print("manual preprocessing before Pipeline will result in data leakage")
        
        return X_train, X_test, y_train, y_test