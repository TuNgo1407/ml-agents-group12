import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

from data_loader import DataLoader
from evaluation_framework import StandardEvaluator


class NeuralNetworkDeveloper:
    def __init__(self, target: str):

        self.target = target
        self.loader = DataLoader()
        self.evaluator = StandardEvaluator()

        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.loader.load_and_split(target=target)

        print(f"NEURAL NETWORK DEVELOPER | Target: {target}")

    def find_best_model(self) -> Pipeline:

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                random_state=42,
                max_iter=3000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30
            ))
        ])

        if self.target == "final_10_percent_reward_mean":
            param_grid = {
                'model__hidden_layer_sizes': [
                    (128, 64),
                    (128, 64, 32),
                    (256, 128, 64)
                ],
                'model__activation': ['relu', 'tanh'],
                'model__alpha': [1e-4, 1e-3, 1e-2],
                'model__learning_rate_init': [1e-3, 5e-4],
                'model__solver': ['adam']
            }
        else:
            param_grid = {
                'model__hidden_layer_sizes': [
                    (64, 32),
                    (128, 64)
                ],
                'model__activation': ['tanh'],
                'model__alpha': [1e-3, 1e-2, 1e-1],
                'model__learning_rate_init': [5e-4, 1e-4],
                'model__solver': ['adam']
            }

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=3,
            n_jobs=-1,
            verbose=0
        )

        search.fit(self.X_train, self.y_train)

        return search.best_estimator_

    def run(self):
        print("\nFinding best neural network model...")

        best_model = self.find_best_model()

        print(f"Model selected: {best_model.get_params()['model'].__class__.__name__}")
        print(f"Best hyperparameters: {best_model.get_params()}")

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
