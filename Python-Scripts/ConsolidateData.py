from EpisodeExtract import get_epi_training_data
from BehaviorFeatures import get_behavior_features
from FindAbsPath import get_absolute_path

"""
Data Aggregation Module 

This module combines behavior configuration features with training performance metrics
"""



"""
    Args:
        run_id (str): run_id
        yaml_abs_path (str): Absolute path to the YAML configuration file
    
    Returns:
        dict: Combined dictionary containing both behavior features and training
              metrics. 
"""

def get_all_data(run_id:str,yaml_abs_path:str,) ->dict:
    try:
        behavior_features = get_behavior_features(yaml_abs_path)
        epi_features = get_epi_training_data(run_id)
        all_features = epi_features | behavior_features 

        return all_features
    except Exception as e:
        raise



if __name__ == "__main__":
    config_abs_path = get_absolute_path("config")
    yaml_path = fr"{config_abs_path}\ppo\3DBall.yaml"
    data = get_all_data("3DBallRunDebug4",yaml_path)
    print(data)