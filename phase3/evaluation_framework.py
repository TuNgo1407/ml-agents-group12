"""
evaluation framework - Pipeline required for safety.
"""

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from typing import Any
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EvaluationMetrics:
    cv_rmse_mean: float
    cv_rmse_std: float
    test_rmse: float
    
    def __str__(self) -> str:
        return (
            f"CV RMSE: {self.cv_rmse_mean:.6f} ± {self.cv_rmse_std:.6f}\n"
            f"Test RMSE: {self.test_rmse:.6f}"
        )


class StandardEvaluator:
    """
    Pipeline mandatory to prevent data leakage.
    
    Method:
    1. 5-fold CV on training data
    2. Fit final model on all training data  
    3. Single evaluation on test data
    
    Selection criterion: Lowest CV RMSE mean
    Research answer: Test RMSE
    """
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    def evaluate(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> EvaluationMetrics:
        """
        Evaluate model using standard method
        
        Args:
            model: sklearn Pipeline (required for safety)
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            EvaluationMetrics
            
        Raises:
            TypeError: If model is not a Pipeline
        """
        if not isinstance(model, Pipeline):
            raise TypeError(
                "Model must be sklearn.pipeline.Pipeline to prevent data leakage.\n"
                "Even if you don't need preprocessing, wrap your model:\n"
                "  Pipeline([('model', YourModel())])\n"
            )
        neg_mse_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=self.kfold,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            error_score='raise'
        )
        cv_rmse_scores = np.sqrt(-neg_mse_scores)
        cv_mean = cv_rmse_scores.mean()
        cv_std = cv_rmse_scores.std()
        
        final_model = clone(model)
        final_model.fit(X_train, y_train)
        
        y_pred = final_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return EvaluationMetrics(
            cv_rmse_mean=cv_mean,
            cv_rmse_std=cv_std,
            test_rmse=test_rmse
        )