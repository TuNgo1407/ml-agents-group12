"""
Linear models dev space.

Freedom: Choose any linear model, regularization, feature engineering.
Constraint: Must use StandardEvaluator for final evaluation.

This script finds:
- Best linear model for final_10_percent_reward_mean (Target 1)
- Best linear model for late_phase_std (Target 2)
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from data_loader import DataLoader
from evaluation_framework import StandardEvaluator
from sklearn.model_selection import GridSearchCV, KFold

class LinearModelDeveloper:
    def __init__(self, target: str):
        """
        Args:
            target: 'final_10_percent_reward_mean' or 'late_phase_std'
        """
        self.target = target
        self.loader = DataLoader()
        self.evaluator = StandardEvaluator()
        
        # Load data
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.loader.load_and_split(target=target)
        
        print(f"\n{'='*70}")
        print(f"LINEAR MODEL DEVELOPER | Target: {target}")
        print(f"{'='*70}")
    
    def find_best_model(self) -> Pipeline:
        """
        IMPLEMENT YOUR APPROACH HERE.
        
        Freedom:
        - Ridge, Lasso, ElasticNet, or any sklearn linear model
        - Polynomial features, interactions, transformations
        - Any hyperparameter tuning method
        
        Requirements:
        - Return sklearn Pipeline
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(include_bias=False)),
            ('model', Ridge(random_state=42)) 
        ])

        # Define hyperparameter grid
        param_grid = {
            'poly__degree': [1, 2],
            'model__alpha': [0.1, 1.0, 10.0, 100.0]
        }

        # Setup cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Run grid search
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        # Fit grid search
        grid.fit(self.X_train, self.y_train)
        pipeline = grid.best_estimator_

        return pipeline
    
    def run(self):
        print("\nFinding best linear model...")
        
        best_model = self.find_best_model()
        
        print(f"Model selected: {type(best_model).__name__}")
        if hasattr(best_model, 'best_params_'):
            print(f"Best hyperparameters: {best_model.best_params_}")
        
        print("\nEvaluating with standard method...")
        metrics = self.evaluator.evaluate(
            best_model,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test
        )
        
        print("RESULTS - RECORD THESE MANUALLY")
        print(f"Target: {self.target}")
        print(f"Family: Linear Model")
        print(f"\n{metrics}")
        
        return best_model, metrics


if __name__ == "__main__":
    print("LINEAR MODEL DEVELOPER - RUNNING BOTH TARGETS")
    
    print("\n### TARGET 1: Final Performance ###")
    dev_target1 = LinearModelDeveloper(target='final_10_percent_reward_mean')
    model1, metrics1 = dev_target1.run()
    
    print("\n### TARGET 2: Training Stability ###")
    dev_target2 = LinearModelDeveloper(target='late_phase_std')
    model2, metrics2 = dev_target2.run()
    
    print("linear model dev - summary")
    print(f"\nTarget 1 (final_10_percent_reward_mean):")
    print(f"  Test RMSE: {metrics1.test_rmse:.6f}")
    print(f"\nTarget 2 (late_phase_std):")
    print(f"  Test RMSE: {metrics2.test_rmse:.6f}")
    print("Record these numbers yo!!!")