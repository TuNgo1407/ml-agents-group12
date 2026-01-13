"""
Neural network dev space

Freedom: Choose any NN architecture, framework, hyperparameters.
Constraint: Must use StandardEvaluator for final evaluation.

This script finds:
- Best NN model for final_10_percent_reward_mean (Target 1)
- Best NN model for late_phase_std (Target 2)
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from data_loader import DataLoader
from evaluation_framework import StandardEvaluator


class NeuralNetworkDeveloper: 
    def __init__(self, target: str):
        """
        Args:
            target: 'final_10_percent_reward_mean' or 'late_phase_std'
        """
        self.target = target
        self.loader = DataLoader()
        self.evaluator = StandardEvaluator()
        
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.loader.load_and_split(target=target)
        
        print(f"NEURAL NETWORK DEVELOPER | Target: {target}")
    
    def find_best_model(self) -> Pipeline:
        """
        IMPLEMENT YOUR APPROACH HERE.
        
        Freedom:
        - Use sklearn MLPRegressor, Keras, PyTorch, etc.
        - Any architecture, activation, optimizer
        - Any hyperparameter tuning method
        - Any regularization, dropout, etc.
        
        Requirements:
        - Return sklearn Pipeline
        - If using Keras/PyTorch, wrap in sklearn-compatible wrapper
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                random_state=42,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1
            )) #simple default
        ])
        
        return pipeline
    
    def run(self):
        print("\nFinding best neural network model...")
        
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
        print(f"Family: Neural Network")
        print(f"\n{metrics}")
        
        return best_model, metrics


if __name__ == "__main__":
    print("NEURAL NETWORK DEVELOPER - RUNNING BOTH TARGETS")
    
    print("\n### TARGET 1: Final Performance ###")
    dev_target1 = NeuralNetworkDeveloper(target='final_10_percent_reward_mean')
    model1, metrics1 = dev_target1.run()
    
    print("\n### TARGET 2: Training Stability ###")
    dev_target2 = NeuralNetworkDeveloper(target='late_phase_std')
    model2, metrics2 = dev_target2.run()
    
    print("neural net dev summary")
    print(f"\nTarget 1 (final_10_percent_reward_mean):")
    print(f"  Test RMSE: {metrics1.test_rmse:.6f}")
    print(f"\nTarget 2 (late_phase_std):")
    print(f"  Test RMSE: {metrics2.test_rmse:.6f}")
    print("record these numbers!")
