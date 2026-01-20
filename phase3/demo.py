import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelResult:
    name: str
    family: str
    target: str
    rmse: float
    mae: float
    r2: float
    predictions: np.ndarray
    actuals: np.ndarray

class DemoEvaluator:
    
    def __init__(self, main_data_path: str = "main.csv", fresh_data_path: str = "demo.csv"):
        self.main_data_path = Path(main_data_path)
        self.fresh_data_path = Path(fresh_data_path)
        self.feature_names = ['learning_rate', 'batch_size', 'buffer_size', 'hidden_units', 'num_layers', 'time_horizon', 'beta', 'gamma', 'lambd', 'max_steps']
        self.targets = ['final_10_percent_reward_mean', 'late_phase_std']
        self._load_data()
        
    def _load_data(self):
        df_main = pd.read_csv(self.main_data_path)
        if 'error_occurred' in df_main.columns:
            df_main = df_main[~df_main['error_occurred'].fillna(False)]
        df_fresh = pd.read_csv(self.fresh_data_path)
        if 'error_occurred' in df_fresh.columns:
            df_fresh = df_fresh[~df_fresh['error_occurred'].fillna(False)]
        self.data = {}
        for target in self.targets:
            df_main_clean = df_main.dropna(subset=self.feature_names + [target])
            X_train = df_main_clean[self.feature_names].values
            y_train = df_main_clean[target].values
            df_fresh_clean = df_fresh.dropna(subset=self.feature_names + [target])
            X_test = df_fresh_clean[self.feature_names].values
            y_test = df_fresh_clean[target].values
            self.data[target] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }
    
    def build_catboost_model(self, target: str) -> Pipeline:
        if target == 'final_10_percent_reward_mean':
            params = {
                'iterations': 1495,
                'depth': 8,
                'learning_rate': 0.011595715593888537,
                'l2_leaf_reg': 1.0266411834830032,
                'border_count': 52,
                'bagging_temperature': 6.260352048800016,
                'min_data_in_leaf': 62,
                'rsm': 0.7104480772620395,
                'random_strength': 0.03414369902168919,
            }
        else:
            params = {
                'iterations': 1384,
                'depth': 8,
                'learning_rate': 0.018707161840618977,
                'l2_leaf_reg': 1.1193282455054083,
                'border_count': 215,
                'bagging_temperature': 5.500347570887909,
                'min_data_in_leaf': 10,
                'rsm': 0.6942648970766546,
                'random_strength': 0.007807196791006507,
            }
        
        params.update({
            'random_state': 42,'verbose': 0,'thread_count': -1
        })
        return Pipeline([
            ('scaler', StandardScaler()),('model', CatBoostRegressor(**params))
        ])
    
    def build_elasticnet_model(self, target: str) -> Pipeline:
        return Pipeline([('scaler', StandardScaler()),('poly', PolynomialFeatures(degree=2, include_bias=False)),('model', ElasticNet(alpha=1.0,l1_ratio=1.0,max_iter=2000,random_state=42))])
    
    def build_mlp_model(self, target: str) -> Pipeline:
        if target == 'final_10_percent_reward_mean':
            params = {
                'hidden_layer_sizes': (256, 128, 64),
                'activation': 'relu',
                'alpha': 0.01,
                'learning_rate_init': 0.001,
            }
        else:
            params = {
                'hidden_layer_sizes': (128, 64),
                'activation': 'tanh',
                'alpha': 0.1,
                'learning_rate_init': 0.0005,
            }
        params.update({
            'solver': 'adam',
            'early_stopping': True,
            'validation_fraction': 0.15,
            'n_iter_no_change': 30,
            'max_iter': 3000,
            'random_state': 42
        })
        
        return Pipeline([('scaler', StandardScaler()),('model', MLPRegressor(**params))])
    
    def evaluate_model(self, pipeline: Pipeline, target: str, name: str, family: str) -> ModelResult:
        data = self.data[target]
        # print(f"\nTraining {name} for {target}...")
        pipeline.fit(data['X_train'], data['y_train'])
        y_pred = pipeline.predict(data['X_test'])
        y_true = data['y_test']
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        # print(f"  RMSE: {rmse:.4f}")
        # print(f"  MAE: {mae:.4f}")
        # print(f"  R²: {r2:.4f}")
        return ModelResult(
            name=name,
            family=family,
            target=target,
            rmse=rmse,
            mae=mae,
            r2=r2,
            predictions=y_pred,
            actuals=y_true
        )
    
    def run_all_models(self) -> Dict[str, List[ModelResult]]:
        results = {target: [] for target in self.targets}
        for target in self.targets:
            # print(f"evaluating for: {target}")
            model = self.build_catboost_model(target)
            result = self.evaluate_model(model, target, "CatBoost", "Tree-based")
            results[target].append(result)
            model = self.build_elasticnet_model(target)
            result = self.evaluate_model(model, target, "ElasticNet", "Linear")
            results[target].append(result)
            model = self.build_mlp_model(target)
            result = self.evaluate_model(model, target, "MLP", "Neural Network")
            results[target].append(result)
        return results
    
    def create_visualizations(self, results: Dict[str, List[ModelResult]]):
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        colors = {'Tree-based': '#2ecc71', 'Linear': '#3498db', 'Neural Network': '#e74c3c'}
        ax1 = fig.add_subplot(gs[0, 0:2]) #rmse
        self._plot_rmse_comparison(ax1, results, colors)
        ax2 = fig.add_subplot(gs[0, 2:4])#r2
        self._plot_r2_comparison(ax2, results, colors)
        ax3 = fig.add_subplot(gs[1, 0:2])#mae
        self._plot_mae_comparison(ax3, results, colors)
        ax4 = fig.add_subplot(gs[1, 2:4])
        self._plot_summary_table(ax4, results)
        target1 = 'final_10_percent_reward_mean'
        for idx, result in enumerate(results[target1]):
            ax = fig.add_subplot(gs[2, idx])
            self._plot_predictions(ax, result, colors[result.family])
        ax8 = fig.add_subplot(gs[2, 3])
        self._plot_winner_highlight(ax8, results)
        
        plt.suptitle('Model Performance Demo', fontsize=20, fontweight='bold', y=0.995)
        plt.savefig('model_performance_demo.png', dpi=300, bbox_inches='tight')
        print("image saved'")
        plt.show()
    
    def _plot_rmse_comparison(self, ax, results, colors):
        data = []
        for target in self.targets:
            for result in results[target]:
                data.append({
                    'Model': result.name,
                    'Target': 'Final Performance' if 'final' in target else 'Training Stability',
                    'RMSE': result.rmse,
                    'Family': result.family
                })
        
        df = pd.DataFrame(data)
        
        x = np.arange(len(self.targets))
        width = 0.25
        
        for idx, model_name in enumerate(['CatBoost', 'ElasticNet', 'MLP']):
            model_data = df[df['Model'] == model_name]
            offsets = x + (idx - 1) * width
            family = model_data.iloc[0]['Family']
            ax.bar(offsets, model_data['RMSE'], width, 
                   label=model_name, color=colors[family], alpha=0.8)
        
        ax.set_xlabel('Target Variable', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
        ax.set_title('Root Mean Squared Error Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Final Performance', 'Training Stability'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_r2_comparison(self, ax, results, colors):
        """Plot R² score comparison."""
        data = []
        for target in self.targets:
            for result in results[target]:
                data.append({
                    'Model': result.name,
                    'Target': 'Final Performance' if 'final' in target else 'Training Stability',
                    'R²': result.r2,
                    'Family': result.family
                })
        
        df = pd.DataFrame(data)
        
        x = np.arange(len(self.targets))
        width = 0.25
        
        for idx, model_name in enumerate(['CatBoost', 'ElasticNet', 'MLP']):
            model_data = df[df['Model'] == model_name]
            offsets = x + (idx - 1) * width
            family = model_data.iloc[0]['Family']
            ax.bar(offsets, model_data['R²'], width, 
                   label=model_name, color=colors[family], alpha=0.8)
        
        ax.set_xlabel('Target Variable', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² Score (Higher is Better)', fontsize=12, fontweight='bold')
        ax.set_title('Coefficient of Determination (R²)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Final Performance', 'Training Stability'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
    
    def _plot_mae_comparison(self, ax, results, colors):
        """Plot MAE comparison."""
        data = []
        for target in self.targets:
            for result in results[target]:
                data.append({
                    'Model': result.name,
                    'Target': 'Final Performance' if 'final' in target else 'Training Stability',
                    'MAE': result.mae,
                    'Family': result.family
                })
        
        df = pd.DataFrame(data)
        
        x = np.arange(len(self.targets))
        width = 0.25
        
        for idx, model_name in enumerate(['CatBoost', 'ElasticNet', 'MLP']):
            model_data = df[df['Model'] == model_name]
            offsets = x + (idx - 1) * width
            family = model_data.iloc[0]['Family']
            ax.bar(offsets, model_data['MAE'], width, 
                   label=model_name, color=colors[family], alpha=0.8)
        
        ax.set_xlabel('Target Variable', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAE (Lower is Better)', fontsize=12, fontweight='bold')
        ax.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Final Performance', 'Training Stability'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_summary_table(self, ax, results):
        ax.axis('tight')
        ax.axis('off')
        table_data = []

        headers = ['Target', 'Model', 'RMSE', 'MAE', 'R²']
        for target in self.targets:
            target_display = 'Final Perf.' if 'final' in target else 'Train Stab.'
            for result in results[target]:
                table_data.append([target_display,result.name,
                    f"{result.rmse:.2f}",
                    f"{result.mae:.2f}",
                    f"{result.r2:.3f}"
                ])
        table = ax.table(cellText=table_data, colLabels=headers,cellLoc='center', loc='center',colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        ax.set_title('Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
    
    def _plot_predictions(self, ax, result: ModelResult, color):
        ax.scatter(result.actuals, result.predictions, alpha=0.5, color=color, s=30)
        min_val = min(result.actuals.min(), result.predictions.min())
        max_val = max(result.actuals.max(), result.predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Values', fontsize=10, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=10, fontweight='bold')
        ax.set_title(f'{result.name}\nRMSE: {result.rmse:.2f}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    def _plot_winner_highlight(self, ax, results):
        ax.axis('off')
        winners_text = "best models! "
        for target in self.targets:
            target_results = results[target]
            best_model = min(target_results, key=lambda x: x.rmse)
            target_display = "Final Performance" if 'final' in target else "Training Stability"
            winners_text += f"{target_display}:\n"
            winners_text += f"  {best_model.name} ({best_model.family})\n"
            winners_text += f"  RMSE: {best_model.rmse:.2f}\n"
            winners_text += f"  R²: {best_model.r2:.3f}\n\n"
        
        ax.text(0.5, 0.5, winners_text, ha='center', va='center',fontsize=12, fontweight='bold',bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3))


def main():
    evaluator = DemoEvaluator()
    results = evaluator.run_all_models()
    evaluator.create_visualizations(results)


if __name__ == "__main__":
    main()