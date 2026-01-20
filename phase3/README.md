# Phase 3: Data Modeling

This phase implements predictive models for ML-Agents training outcomes. All models predict two regression targets derived from training logs: final reward performance and late-phase stability. Emphasis is placed on reproducibility through standardized data handling, evaluation protocols, and hyperparameter management.

## Overview

### Models Implemented

- **Linear Models** (`linear_model_developer.py`): Ridge, Lasso, ElasticNet with polynomial features
- **Neural Networks** (`neural_network_developer.py`): Multi-layer perceptrons with target-specific architectures
- **Tree-based Models** (`tree_based_developer.py`): XGBoost, CatBoost, LightGBM, RandomForest, ExtraTrees with Optuna optimization
- **Demo** (`demo.py`): Evaluation framework for pre-trained models on fresh data

### Data Requirements

- **Training Data**: `main.csv` - contains 10 hyperparameters and 2 regression targets
- **Test Data**: `demo.csv` - fresh data for out-of-sample evaluation

**Features** (10 hyperparameters):
```
learning_rate, batch_size, buffer_size, hidden_units, num_layers, 
time_horizon, beta, gamma, lambd, max_steps
```

**Targets** (regression):
```
final_10_percent_reward_mean - mean reward in final 10% of training
late_phase_std - standard deviation of reward during late training phase
```

## Usage

### Running Model Development

Each developer script finds the optimal model for a given target using different optimization strategies.

#### Linear Models
```bash
python linear_model_developer.py
```
Performs grid search over Ridge, Lasso, and ElasticNet with polynomial feature engineering. Outputs best model CV RMSE and test metrics.

#### Neural Networks
```bash
python neural_network_developer.py
```
Optimizes MLP architectures independently per target using 3-fold CV grid search. Models include early stopping and validation-based regularization.

#### Tree-based Models
```bash
python tree_based_developer.py
```
Orchestrates Optuna-based hyperparameter optimization for 5 tree families independently per target. Searches over 100+ hyperparameter combinations per family using TPE sampler.

### Running Evaluation on Fresh Data

```bash
python demo.py
```

Evaluates pre-trained models (CatBoost, ElasticNet, MLP) on fresh test data (`demo.csv`) from the main training set. Outputs RMSE, MAE, and R² scores without retraining.

## Architecture & Safety

### Data Leakage Prevention

All models enforce preprocessing inside sklearn Pipelines to prevent data leakage:
- Scaling happens **only on training data**, transformations applied to test data
- Feature engineering (polynomial, etc.) fitted on training data only
- No preprocessing applied in `DataLoader` - all handled within Pipelines

### Consistent Evaluation

`StandardEvaluator` enforces a uniform evaluation protocol:
1. **5-fold cross-validation** on training data (selection criterion)
2. **Full training** on entire training set
3. **Test evaluation** on held-out test set (final metric)

All developers use `random_state=42` for reproducibility.

### Fixed Train/Test Split

`DataLoader` maintains a fixed 80/20 train/test split with `random_state=42`. All developers use the same split for fair comparison.

## Results Recording

Model developers output optimized hyperparameters and CV scores to console. **Results must be recorded manually** in persistent storage. For standardized tracking:
- Model family and architecture
- Optimized hyperparameters
- CV RMSE mean and std
- Test RMSE, MAE, R²

The demo script outputs metrics for pre-trained models and can serve as a reference for expected performance.

## Dependencies

```
pandas
numpy
scikit-learn
optuna
xgboost
catboost
lightgbm
matplotlib
seaborn
```

Install via:
```bash
pip install -r ../requirements.txt
```

## File Manifest

| File | Purpose |
|------|---------|
| `data_loader.py` | Loads CSV data with fixed train/test split, no preprocessing |
| `evaluation_framework.py` | StandardEvaluator - enforces Pipeline-based evaluation with CV |
| `linear_model_developer.py` | Finds optimal linear model per target via grid search |
| `neural_network_developer.py` | Finds optimal MLP architecture per target |
| `tree_based_developer.py` | Finds optimal tree model per target using Optuna |
| `demo.py` | Evaluates pre-trained models on fresh data |
| `main.csv` | Training dataset |
| `demo.csv` | Fresh test dataset |

## Reproducibility Notes

- All random seeds fixed to 42
- Models use same train/test split across all families
- Feature engineering and scaling encapsulated in Pipelines
- Hyperparameter grids and Optuna search spaces defined in code (easily modified)
- No environment-dependent operations (paths, random number generation)

To reproduce results with different hyperparameter spaces, modify the grid definitions or Optuna search space in the respective developer scripts.
