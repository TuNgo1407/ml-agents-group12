"""
Tree-based models dev space.

Freedom: Choose any tree algorithm, any hyperparameters, any tuning method.
Constraint: Must use StandardEvaluator for final evaluation.

This script finds:
- Best tree model for final_10_percent_reward_mean (Target 1)
- Best tree model for late_phase_std (Target 2)

"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from data_loader import DataLoader
from evaluation_framework import StandardEvaluator


class TreeDeveloper:
    
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
        print(f"TREE-BASED DEVELOPER | Target: {target}")
        print(f"{'='*70}")
    
    def find_best_model(self) -> Pipeline:
        """
        IMPLEMENT YOUR APPROACH HERE.
        
        You have complete freedom:
        - Choose any tree algorithm (RandomForest, XGBoost, LightGBM, CatBoost, etc.)
        - Use any hyperparameter tuning (GridSearch, RandomSearch, Bayesian, etc.)
        - Apply any preprocessing or feature engineering
        - Use any validation strategy for tuning
        
        Requirements:
        - Return a sklearn Pipeline (recommended for safety)
        - Pipeline should be unfitted (evaluator will handle fitting)

        """
        
        # OPTION 1: Simple RandomForest (baseline)
        # pipeline = Pipeline([
        #     ('model', RandomForestRegressor(
        #         n_estimators=100,
        #         max_depth=10,
        #         random_state=42,
        #         n_jobs=-1
        #     ))
        # ])
        # return pipeline
        
        # OPTION 2: GridSearch for hyperparameter tuning
        # base_pipeline = Pipeline([
        #     ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
        # ])
        # 
        # param_grid = {
        #     'model__n_estimators': [50, 100, 200],
        #     'model__max_depth': [5, 10, 15, None],
        #     'model__min_samples_split': [2, 5, 10]
        # }
        # 
        # search = GridSearchCV(
        #     base_pipeline,
        #     param_grid,
        #     cv=5,
        #     scoring='neg_root_mean_squared_error',
        #     n_jobs=-1,
        #     verbose=1
        # )
        # 
        # return search
        # OPTION 3: Compare multiple algorithms
        # from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        # 
        # # Create multiple candidates
        # candidates = [
        #     Pipeline([('model', RandomForestRegressor(n_estimators=100, random_state=42))]),
        #     Pipeline([('model', GradientBoostingRegressor(n_estimators=100, random_state=42))])
        # ]
        # 
        # # Find best using cross-validation
        # best_score = float('inf')
        # best_model = None
        # 
        # for candidate in candidates:
        #     from sklearn.model_selection import cross_val_score
        #     scores = cross_val_score(candidate, self.X_train, self.y_train,
        #                             cv=5, scoring='neg_mean_squared_error')
        #     rmse = np.sqrt(-scores.mean())
        #     if rmse < best_score:
        #         best_score = rmse
        #         best_model = candidate
        # 
        # return best_model
        
        # DEFAULT: Simple baseline
        pipeline = Pipeline([
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        return pipeline
    
    def run(self):
        """
        
        Returns:
            (model, metrics)
        """
        print("\nFinding best tree model...")
        
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
        
        print(f"\n{'='*70}")
        print("RESULTS - RECORD THESE MANUALLY")
        print(f"{'='*70}")
        print(f"Target: {self.target}")
        print(f"Family: Tree-based")
        print(f"\n{metrics}")
        print(f"\n{'='*70}")
        
        return best_model, metrics


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TREE-BASED DEVELOPER - RUNNING BOTH TARGETS")
    print("="*70)
    
    print("\n### TARGET 1: Final Performance ###")
    dev_target1 = TreeDeveloper(target='final_10_percent_reward_mean')
    model1, metrics1 = dev_target1.run()
    
    print("\n### TARGET 2: Training Stability ###")
    dev_target2 = TreeDeveloper(target='late_phase_std')
    model2, metrics2 = dev_target2.run()
    
    print("summary - tree based")
    print(f"\nTarget 1 (final_10_percent_reward_mean):")
    print(f"  Test RMSE: {metrics1.test_rmse:.6f}")
    print(f"\nTarget 2 (late_phase_std):")
    print(f"  Test RMSE: {metrics2.test_rmse:.6f}")
    print("record these!")
