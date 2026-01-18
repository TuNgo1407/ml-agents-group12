"""
Linear models dev space.

Freedom: Choose any linear model, regularization, feature engineering.
Constraint: Must use StandardEvaluator for final evaluation.

This script finds:
- Best linear model for final_10_percent_reward_mean (Target 1)
- Best linear model for late_phase_std (Target 2)
"""

from typing import Tuple
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
        Finds the best linear model among Ridge, ElasticNet, and Lasso
        Each model is trained with hyperparameter tuning via grid search.
        Returns the best model pipeline.
        """

        print("Training Ridge... ")
        best_ridge, ridge_score = self.ridge()
        print(f"Ridge best score (neg MSE): {ridge_score:.4f}")

        print("Training ElasticNet... ")
        best_elastic_net, elastic_net_score = self.elastic_net()
        print(f"ElasticNet best score (neg MSE): {elastic_net_score:.4f}")

        print("Training Lasso... ")
        best_lasso, lasso_score = self.Lasso()
        print(f"Lasso best score (neg MSE): {lasso_score:.4f}")

        scores = {
            'Ridge': ridge_score,
            'ElasticNet': elastic_net_score,
            'Lasso': lasso_score
        }

        best_model = max(scores, key=scores.get)
        print(f"Best model selected: {best_model} with score {scores[best_model]:.4f}")

        if best_model == 'Ridge':
            pipeline = best_ridge
        elif best_model == 'ElasticNet':
            pipeline = best_elastic_net
        else:
            pipeline = best_lasso

        return pipeline
    
    def ridge(self) -> Tuple[Pipeline, float]:
         # Runs grid search for Ridge model
        # Define pipeline
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
            verbose=0
        )

        # Fit grid search
        grid.fit(self.X_train, self.y_train)

        return grid.best_estimator_, grid.best_score_
    
    def elastic_net(self) -> Tuple[Pipeline, float]:
        # Runs grid search for ElasticNet model
        # Define pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(include_bias=False)),
            ('model', ElasticNet(random_state=42, max_iter=2000)) 
        ])

        # Define hyperparameter grid
        param_grid = {
            'poly__degree': [1, 2],
            'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'model__l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.99, 1.0]
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
            verbose=0
        )

        # Fit grid search
        grid.fit(self.X_train, self.y_train)

        return grid.best_estimator_, grid.best_score_
    
    def Lasso(self) -> Tuple[Pipeline, float]:
         # Runs grid search for Lasso model
        # Define pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(include_bias=False)),
            ('model', Lasso(random_state=42, max_iter=2000)) 
        ])

        # Define hyperparameter grid
        param_grid = {
            'poly__degree': [1, 2],
            'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
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
            verbose=0
        )

        # Fit grid search
        grid.fit(self.X_train, self.y_train)

        return grid.best_estimator_, grid.best_score_


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