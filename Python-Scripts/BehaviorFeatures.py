import yaml 
import os


"""
This module provides functionality to extract specific hyperparameters and 
configuration settings from YAML configuration files.
"""

BEHAVIOR_FEATURE_PATHS = {
    "learning_rate": "hyperparameters.learning_rate"
    ,"batch_size": "hyperparameters.batch_size"
    ,"buffer_size": "hyperparameters.buffer_size"
    ,"beta": "hyperparameters.beta"
    ,"lambd": "hyperparameters.lambd"
    ,"hidden_units": "network_settings.hidden_units"
    ,"num_layers": "network_settings.num_layers"
    ,"time_horizon": "time_horizon"
    ,"gamma": "reward_signals.extrinsic.gamma"
    ,"max_steps": "max_steps"
}


"""
    Extracts specified behavior features from a YAML configuration file.
    
    This function reads a YAML configuration file, validates its structure,
    and extracts predefined hyperparameters and network settings for a specific
    behavior
    
    Args:
        yaml_abs_path (str): Absolute path to the YAML configuration file
        game (str, optional): Name of the behavior to extract features from. 
                            Defaults to "3DBall".
    
    Returns:
        dict: Dictionary containing the extracted features with keys from 
              BEHAVIOR_FEATURE_PATHS and their corresponding values.
    
    Example:
        >>> features = get_behavior_features("/path/to/config.yaml", "3DBall")
        >>> print(features["learning_rate"])
        0.0003
"""
def get_behavior_features(yaml_abs_path:str, game:str="3DBall") -> dict:
    try:
        if not os.path.exists(yaml_abs_path):
            raise FileNotFoundError(f"Yaml file not found: {yaml_abs_path}")
        
        with open(yaml_abs_path,"r") as file:
            data = yaml.safe_load(file)

        if data is None:
            raise ValueError(f"Yaml file is empty")
        
        if "behaviors" not in data:
            raise KeyError("No behaviors key")
        
        if game not in data["behaviors"]:
            raise KeyError(f"Not found behavior {game}")
        
        behavior_data = data["behaviors"][game]

        behavior_feature_dict = {}
        for metric, path in BEHAVIOR_FEATURE_PATHS.items():
            behavior_feature_dict[metric] = _get_by_dot_path(behavior_data, path)

        return behavior_feature_dict
    except Exception as e:
        print(f"Error when extract yaml file: {e}")

def _get_by_dot_path(data:dict, path:str):
    current_data = data
    keys = path.split(".")

    for key in keys:
        current_data = current_data[key]
    return current_data



