import random
import yaml
from datetime import datetime
from typing import Dict, Any
import os
from FindAbsPath import get_absolute_path

class ConfigGenerator:
    """Generates randomized hyperparameter configurations for 3DBall"""

    PARAM_RANGES = {
        "learning_rate": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
        "batch_size": [32, 64, 128, 256, 512, 1024, 2048],
        "buffer_size": [1024, 2048, 4096, 8192, 16384, 32768],
        "hidden_units": [32, 64, 128, 256, 512],
        "num_layers": [1, 2, 3],
        "time_horizon": [32, 64, 128, 256],
        "beta": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        "gamma": [0.9, 0.92, 0.95, 0.97, 0.99, 0.995, 0.999],
        "lambd": [0.9, 0.92, 0.95, 0.97, 0.99],
        "max_steps": [250000, 500000, 750000, 1000000]
    }

    @classmethod
    def generate_config(cls) -> Dict[str, Any]:
        """Generate a random configuration sampling from predefined ranges"""
        config = {}

        for param, values in cls.PARAM_RANGES.items():
            config[param] = random.choice(values)

        return config

    @classmethod
    def create_run_id(cls) -> str:
        """Generate unique run ID with timestamp and random suffix"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
        return f"3DBall_{timestamp}_{random_suffix}"

    @classmethod
    def update_yaml_config(cls, base_config_path: str, new_config: Dict[str, Any], output_path: str):
        """Update base YAML config with new hyperparameters"""
        with open(base_config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        behavior_name = "3DBall"
        if behavior_name in config_data.get("behaviors", {}):
            behavior = config_data["behaviors"][behavior_name]

            param_mapping = {
                "learning_rate": ["hyperparameters", "learning_rate"],
                "batch_size": ["hyperparameters", "batch_size"],
                "buffer_size": ["hyperparameters", "buffer_size"],
                "beta": ["hyperparameters", "beta"],
                "lambd": ["hyperparameters", "lambd"],
                "hidden_units": ["network_settings", "hidden_units"],
                "num_layers": ["network_settings", "num_layers"],
                "time_horizon": ["time_horizon"],
                "gamma": ["reward_signals", "extrinsic", "gamma"],
                "max_steps": ["max_steps"]
            }

            for param, value in new_config.items():
                if param in param_mapping:
                    current = behavior
                    path = param_mapping[param]

                    for key in path[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]

                    current[path[-1]] = value
            behavior["behavioral_cloning"] = {
                "demo_path": "/Users/holom/PycharmProjects/ml-agents-group12/Project/Assets/ML-Agents/Examples/3DBall/Demos/Expert3DBall.demo",
                "strength": 0.5,
                "steps": 100000
            }


        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """Validate that the generated config is reasonable"""
        if config['batch_size'] > config['buffer_size']:
            return False

        if config['time_horizon'] > config['buffer_size']:
            return False

        return True

    @classmethod
    def generate_valid_config(cls, max_attempts: int = 10) -> Dict[str, Any]:
        """Generate a valid configuration with constraints"""
        for _ in range(max_attempts):
            config = cls.generate_config()
            if cls.validate_config(config):
                return config
        return config
