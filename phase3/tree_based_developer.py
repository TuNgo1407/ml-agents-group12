"""
Tree-based models development using Optuna hyperparameter optimization.

Architecture:
- Optimize 5 model families: XGBoost, CatBoost, LightGBM, RandomForest, ExtraTrees
- Optimize both targets: final_10_percent_reward_mean, late_phase_std
- Use StandardEvaluator for evaluation (Pipeline enforced)
- Hyperparameter ranges defined in HyperSpace (easily swappable)

Usage:
    python tree_based_developer.py
"""
import numpy as np
from typing import Any, Dict, List
from dataclasses import dataclass
from enum import Enum
import optuna
from optuna.samplers import TPESampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from data_loader import DataLoader
from evaluation_framework import StandardEvaluator


# We define the search space below, it can be easily changed.
@dataclass
class ParamRange:
    min_val: float
    max_val: float
    log_scale: bool = False
    
@dataclass
class IntParamRange:
    min_val: int
    max_val: int
    log_scale: bool = False
    
@dataclass
class CategoricalParam:
    choices: List[Any]


class HyperSpace:
    XGBOOST = {
        'n_estimators': IntParamRange(100, 1500),
        'max_depth': IntParamRange(3, 12),
        'learning_rate': ParamRange(0.001, 0.5, log_scale=True),
        'subsample': ParamRange(0.5, 1.0),
        'colsample_bytree': ParamRange(0.3, 1.0),
        'colsample_bylevel': ParamRange(0.3, 1.0),
        'colsample_bynode': ParamRange(0.3, 1.0),
        'min_child_weight': IntParamRange(1, 20),
        'gamma': ParamRange(0.0, 10.0),
        'reg_alpha': ParamRange(1e-8, 100.0, log_scale=True),
        'reg_lambda': ParamRange(1e-8, 100.0, log_scale=True),
        'max_delta_step': IntParamRange(0, 10),
        'grow_policy': CategoricalParam(['depthwise', 'lossguide']),
    }
    CATBOOST = {
        'iterations': IntParamRange(100, 1500),
        'depth': IntParamRange(3, 10),
        'learning_rate': ParamRange(0.001, 0.5, log_scale=True),
        'l2_leaf_reg': ParamRange(1.0, 30.0),
        'border_count': IntParamRange(32, 255),
        'bagging_temperature': ParamRange(0.0, 10.0),
        'min_data_in_leaf': IntParamRange(1, 100),
        'rsm': ParamRange(0.3, 1.0),
        'random_strength': ParamRange(0.0, 10.0),
    }
    LIGHTGBM = {
        'n_estimators': IntParamRange(100, 1500),
        'max_depth': IntParamRange(3, 12),
        'learning_rate': ParamRange(0.001, 0.5, log_scale=True),
        'num_leaves': IntParamRange(20, 255),
        'subsample': ParamRange(0.5, 1.0),
        'colsample_bytree': ParamRange(0.3, 1.0),
        'min_child_samples': IntParamRange(5, 200),
        'min_child_weight': ParamRange(1e-5, 10.0, log_scale=True),
        'reg_alpha': ParamRange(1e-8, 100.0, log_scale=True),
        'reg_lambda': ParamRange(1e-8, 100.0, log_scale=True),
        'subsample_freq': IntParamRange(0, 10),
        'boosting_type': CategoricalParam(['gbdt', 'dart', 'goss']),
    }
    RANDOM_FOREST = {
        'n_estimators': IntParamRange(100, 1000),
        'max_depth': IntParamRange(3, 20),
        'min_samples_split': IntParamRange(2, 30),
        'min_samples_leaf': IntParamRange(1, 20),
        'max_features': CategoricalParam(['sqrt', 'log2', 0.3, 0.5, 0.7, 0.9]),
        'min_impurity_decrease': ParamRange(0.0, 0.1),
        'ccp_alpha': ParamRange(0.0, 0.2),
        'bootstrap': CategoricalParam([True, False]),
        'criterion': CategoricalParam(['squared_error', 'absolute_error', 'friedman_mse']),
    }
    EXTRA_TREES = {
        'n_estimators': IntParamRange(100, 1000),
        'max_depth': IntParamRange(3, 20),
        'min_samples_split': IntParamRange(2, 30),
        'min_samples_leaf': IntParamRange(1, 20),
        'max_features': CategoricalParam(['sqrt', 'log2', 0.3, 0.5, 0.7, 0.9]),
        'min_impurity_decrease': ParamRange(0.0, 0.1),
        'ccp_alpha': ParamRange(0.0, 0.2),
        'bootstrap': CategoricalParam([True, False]),
        'criterion': CategoricalParam(['squared_error', 'absolute_error', 'friedman_mse']),
    }

class TreeModelTuner:
    """
    Optuna-based hyperparameter tuner for tree-based models.
    Optimizes all model families using cross-validation on training data.
    Returns best model evaluated on test set.
    """
    MODEL_REGISTRY = {
        'xgboost': XGBRegressor,
        'catboost': CatBoostRegressor,
        'lightgbm': LGBMRegressor,
        'random_forest': RandomForestRegressor,
        'extra_trees': ExtraTreesRegressor,
    }
    HYPERPARAM_REGISTRY = {
        'xgboost': HyperSpace.XGBOOST,
        'catboost': HyperSpace.CATBOOST,
        'lightgbm': HyperSpace.LIGHTGBM,
        'random_forest': HyperSpace.RANDOM_FOREST,
        'extra_trees': HyperSpace.EXTRA_TREES,
    }
    
    def __init__(self, target_name: str, random_state: int = 42):
        """
        Initialize tuner for specific target.
        
        Args:
            target_name: 'final_10_percent_reward_mean' or 'late_phase_std'
            random_state: Random seed for reproducibility
        """
        self.target_name = target_name
        self.random_state = random_state
        
        loader = DataLoader()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            loader.load_and_split(target=target_name)
        self.evaluator = StandardEvaluator(random_state=random_state)
        print("-----------------------------------------------------------")
        print(f"TreeModelTuner initialized")
        print(f"Target: {target_name}")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print("-----------------------------------------------------------")
    
    def _instantiate_model(self, trial: optuna.Trial, model_name: str) -> Any:
        """
        instantiate model with hyperparameters from trial.
        Args:
            trial: Optuna trial object
            model_name: One of: 'xgboost', 'catboost', 'lightgbm', 
                       'random_forest', 'extra_trees', or any other
        Returns:
            Instantiated model object with suggested hyperparameters
        """
        model_class = self.MODEL_REGISTRY[model_name]
        param_space = self.HYPERPARAM_REGISTRY[model_name]
        params = {}
        for param_name, param_range in param_space.items():
            if isinstance(param_range, CategoricalParam):
                params[param_name] = trial.suggest_categorical(
                    f"{model_name}_{param_name}",
                    param_range.choices
                )
            elif isinstance(param_range, IntParamRange):
                params[param_name] = trial.suggest_int(
                    f"{model_name}_{param_name}",
                    param_range.min_val,
                    param_range.max_val,
                    log=param_range.log_scale
                )
            else:
                params[param_name] = trial.suggest_float(
                    f"{model_name}_{param_name}",
                    param_range.min_val,
                    param_range.max_val,
                    log=param_range.log_scale
                )
        if model_name == 'xgboost':
            params['random_state'] = self.random_state
            params['n_jobs'] = -1
            params['verbosity'] = 0
            
        elif model_name == 'catboost':
            params['random_state'] = self.random_state
            params['verbose'] = 0
            params['thread_count'] = -1
            
        elif model_name == 'lightgbm':

            params['random_state'] = self.random_state
            params['n_jobs'] = -1
            params['verbose'] = -1
            
            #lightGBM constraint- GOSS doesn't support bagging
            if params.get('boosting_type') == 'goss':
                params['subsample'] = 1.0
                params['subsample_freq'] = 0
        else:# sklearn models
            params['random_state'] = self.random_state
            params['n_jobs'] = -1
            if 'max_features' in params:
                max_features_val = params['max_features']

                if isinstance(max_features_val, float):
                    params['max_features'] = float(max_features_val)
        return model_class(**params)
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Cross-validation RMSE
        """
        try:
            model_family = trial.suggest_categorical(
                'model_family',list(self.MODEL_REGISTRY.keys())
            )
            model = self._instantiate_model(trial, model_family)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            metrics = self.evaluator.evaluate(
                pipeline,
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test
            )
            trial.set_user_attr('test_rmse', metrics.test_rmse)
            trial.set_user_attr('cv_std', metrics.cv_rmse_std)
            trial.set_user_attr('model_family', model_family)
            
            
            return metrics.cv_rmse_mean
            
        except Exception as e:
            if "Cannot use bagging in GOSS" in str(e):
                return float('inf')
            else:
                raise
    
    def tune(self, n_trials: int = 1500, show_progress: bool = True) -> Pipeline:
        """
        Run Optuna study to find best model.
        Args:
            n_trials: Number of trials to run 
            show_progress: Whether to show progress bar
        
        Returns:
            Best pipeline
        """
        print(f"Starting optimization with expanded hyperspace: {n_trials} trials")
        print(f"Model families: {list(self.MODEL_REGISTRY.keys())}\n")
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(
                seed=self.random_state,
                n_startup_trials=100,  # may increase for better exploration
                multivariate=True,
                warn_independent_sampling=False #DO NOT set this true unless you want to flood your terminal
            )
        )
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=show_progress,


            catch=(ValueError)
        )
        best_trial = study.best_trial
        best_family = best_trial.params['model_family']
        best_model = self._instantiate_model(best_trial, best_family)
        best_pipeline = Pipeline([('scaler', StandardScaler()),('model', best_model)])
        
        best_pipeline.fit(self.X_train, self.y_train)
        self._print_results(study, best_trial, best_family)
        
        return best_pipeline
    
    def _print_results(
        self,
        study: optuna.Study,
        best_trial: optuna.Trial,
        best_family: str
    ):
        print("-----------------------------------------------------------")
        print(f"winner - {self.target_name}")
        print("-----------------------------------------------------------")
        print(f"Model Family: {best_family}")
        print(f"CV RMSE: {best_trial.value:.6f} ± {best_trial.user_attrs['cv_std']:.6f}")
        print(f"Test RMSE: {best_trial.user_attrs['test_rmse']:.6f}")
        family_counts = {}
        for trial in study.trials:
            family = trial.user_attrs.get('model_family', 'unknown')
            family_counts[family] = family_counts.get(family, 0) + 1
        print(f"\nStudy Statistics:")
        print(f"Total Trials: {len(study.trials)}")
        for family, count in sorted(family_counts.items()):
            percentage = (count / len(study.trials)) * 100
            print(f"  {family}: {count} trials ({percentage:.1f}%)")
        print(f"\nBest Hyperparameters:")
        model_params = {
            k.replace(f"{best_family}_", ""): v 
            for k, v in best_trial.params.items() 
            if k != 'model_family'
        }
        for param, value in sorted(model_params.items()):
            print(f"  {param}: {value}")
        
        print("-----------------------------------------------------------")

def main():
    
    targets = ['final_10_percent_reward_mean', 'late_phase_std']
    n_trials = 2200
    
    results = {}
    
    print("-----------------------------------------------------------")
    print(f"Trials per target: {n_trials}")
    print(f"Targets: {targets}")
    
    for target in targets:
        print("-----------------------------------------------------------")
        print(f"# TARGET: {target}")
        print("-----------------------------------------------------------")
        tuner = TreeModelTuner(target_name=target)
        best_model = tuner.tune(n_trials=n_trials)
        results[target] = best_model
    print("-----------------------------------------------------------")
    
    return results


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    results = main()