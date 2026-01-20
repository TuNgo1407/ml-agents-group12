# ML-Agents RL Training Data Modeling

This is a fork of the [official Unity ML-Agents repository](https://github.com/Unity-Technologies/ml-agents) customized for data collection and predictive modeling of deep reinforcement learning training runs.

## Project Objective

Collect empirical data from RL training sessions in Unity, then train machine learning models to predict training outcomes (final reward performance and training stability) from hyperparameter configurations. This enables rapid assessment of hyperparameter choices without running full training sessions.

## Quick Start

### Prerequisites

- Git
- Conda
- Unity Hub + Unity Editor 2023.2.12f1
- Python 3.10.11

### Installation

```bash
# Clone repository
git clone https://github.com/TuNgo1407/ml-agents-group12.git
cd ml-agents-group12

# Create Python environment
conda create -n mlagents python=3.10.11 && conda activate mlagents

# Install ML-Agents and dependencies
pip install mlagents==1.1.0
pip install -r requirements.txt

# Windows only: install PyTorch
pip3 install torch~=1.13.1 -f https://download.pytorch.org/whl/torch_stable.html

# Create results directory
mkdir results
```

For detailed installation, see [docs/Installation.md](docs/Installation.md).

## Project Structure

```
.
├── Python-Scripts/           # Data collection pipeline
│   ├── RunMlagents.py       # Execute training with convergence detection
│   ├── StartTraining.py      # Automated training orchestration
│   ├── ConfigGenerator.py    # Hyperparameter configuration creation
│   ├── ConsolidateData.py    # Extract training metrics to CSV
│   ├── SupabaseAccess.py     # Database integration
│   └── Analysis/             # Post-training analysis tools
│       ├── PullData.py       # Fetch training data from Supabase
│       └── Analysis.ipynb    # Data visualization and exploration
│
├── phase3/                   # Predictive modeling
│   ├── data_loader.py        # Load and split training data
│   ├── evaluation_framework.py # Standard evaluation protocol (5-fold CV)
│   ├── linear_model_developer.py     # Ridge/Lasso/ElasticNet optimization
│   ├── neural_network_developer.py   # MLP architecture optimization
│   ├── tree_based_developer.py       # XGBoost/CatBoost/LightGBM optimization
│   ├── demo.py               # Evaluate pre-trained models on fresh data
│   ├── main.csv              # Aggregated training dataset
│   └── demo.csv              # Fresh test dataset
│
├── com.unity.ml-agents/      # Official ML-Agents package
├── com.unity.ml-agents.extensions/ # Extended functionality
├── Project/                  # Unity 3DBall training environment
├── config/                   # ML-Agents training configs (ppo, sac, etc)
├── docs/                     # Complete documentation
└── requirements.txt          # Python dependencies
```

## Workflow

### 1. Data Collection (Python-Scripts/)

Automated training pipeline that generates hyperparameter configurations and runs RL training in Unity.

```bash
cd Python-Scripts
python StartTraining.py
```

**What happens:**
1. Generates random hyperparameter configuration
2. Launches Unity environment with generated config
3. Monitors training for convergence
4. Extracts training metrics (reward, loss, etc.)
5. Stores results in Supabase database and CSV

**Key Features:**
- Configuration-driven (no code changes between runs)
- Early stopping on convergence detection
- Cross-platform support (Windows, macOS, Linux)
- Database integration for distributed collection
- Automatic data consolidation to CSV

**Configuration:** Modify [Python-Scripts/ConfigGenerator.py](Python-Scripts/ConfigGenerator.py) to adjust hyperparameter ranges and generation strategy.

### 2. Data Analysis (Python-Scripts/Analysis/)

Explore and visualize collected training data.

```bash
cd Python-Scripts/Analysis
jupyter notebook Analysis.ipynb
```

**Fetches data** from Supabase and generates exploratory plots showing hyperparameter-outcome relationships.

### 3. Predictive Modeling (phase3/)

Train ML models to predict training outcomes from hyperparameters.

```bash
cd phase3

# Optimize linear models
python linear_model_developer.py

# Optimize neural networks
python neural_network_developer.py

# Optimize tree-based models
python tree_based_developer.py

# Evaluate pre-trained models on fresh data
python demo.py
```

**Targets predicted:**
- `final_10_percent_reward_mean` - mean reward in final 10% of training
- `late_phase_std` - reward stability during late training phase

**Features used** (10 hyperparameters):
```
learning_rate, batch_size, buffer_size, hidden_units, num_layers,
time_horizon, beta, gamma, lambd, max_steps
```

**Model families:**
- Linear: Ridge, Lasso, ElasticNet with polynomial features
- Neural Networks: MLPs with target-specific architectures
- Tree-based: XGBoost, CatBoost, LightGBM, RandomForest, ExtraTrees

**Safety guarantees:**
- Fixed random seeds (42) across all models
- Consistent 80/20 train/test split
- All preprocessing inside sklearn Pipelines (prevents data leakage)
- Standardized evaluation: 5-fold CV → full training → test evaluation

For detailed modeling documentation, see [phase3/README.md](phase3/README.md).

## Custom Additions

This fork extends the official ML-Agents with two custom directories:

### Python-Scripts

Adds automated data collection infrastructure:
- Training orchestration with convergence detection
- Configuration management (no hardcoded parameters)
- Database integration (Supabase)
- Metrics extraction and consolidation
- Analysis notebooks

See [Python-Scripts/README.md](Python-Scripts/README.md) for complete documentation.

### phase3

Adds predictive modeling framework:
- Standardized data loading and splitting
- Evaluation framework (enforces reproducible CV protocol)
- Model developers for 3 architecture families
- Demo evaluation script for pre-trained models

See [phase3/README.md](phase3/README.md) for complete documentation.

## Reproducibility

All experiments are reproducible without code changes:

**Data Collection:**
- Use configuration files to specify hyperparameter ranges
- Modify [ConfigGenerator.py](Python-Scripts/ConfigGenerator.py) for different search spaces
- All runs tracked with unique `run_id` in database and filesystem

**Modeling:**
- Hyperparameter grids defined in code (easily modified)
- Fixed random seeds (42) throughout
- Results depend only on input CSV and hyperparameter specifications
- Output metrics can be recorded and compared

## Dependencies

Core:
- `mlagents==1.1.0` - Unity RL training framework
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - ML utilities and preprocessing
- `tensorflow` - Neural network training (required by mlagents)

Modeling:
- `xgboost`, `catboost`, `lightgbm` - Tree-based models
- `optuna` - Hyperparameter optimization

Database:
- `supabase`, `python-dotenv` - Remote data storage

Visualization:
- `matplotlib`, `seaborn` - Plotting

## References

- [Official ML-Agents Repository](https://github.com/Unity-Technologies/ml-agents)
- [Dennis Soemers Fork](https://github.com/DennisSoemers/ml-agents/tree/fix-numpy-release-21-branch)
- [Unity ML-Agents Documentation](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Readme.md)

## Directory Guide

| Directory | Purpose |
|-----------|---------|
| `Python-Scripts/` | Data collection and training orchestration |
| `phase3/` | Predictive modeling and evaluation |
| `com.unity.ml-agents/` | Core ML-Agents package |
| `Project/` | Unity environment (3DBall) |
| `config/` | Training configurations (PPO, SAC, etc) |
| `docs/` | Extended documentation |
| `results/` | Training outputs (created after first run) |

## Next Steps

1. **Set up environment**: Follow Installation section above
2. **Run a training**: Execute `python Python-Scripts/StartTraining.py` 
3. **Collect data**: Repeat step 2 multiple times to build dataset
4. **Analyze results**: Use notebooks in `Python-Scripts/Analysis/`
5. **Train models**: Run developers in `phase3/` on aggregated `main.csv`
6. **Evaluate**: Use `demo.py` to predict outcomes on fresh data

## Authors

Group 12 - Project 2.1: AI and Machine Learning in the Unity Game Engine
