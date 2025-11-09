import os
import tempfile
from SupabaseAccess import upsert_training_table, check_run_id_exists
from ConsolidateData import get_all_data
from RunMlagents import run_mlagents
from FindAbsPath import get_absolute_path
from ConfigGenerator import ConfigGenerator
import time
from datetime import datetime

def main():
    """Automatically generate config, run training, and store results"""
    
    run_id = ConfigGenerator.create_run_id()
    new_config = ConfigGenerator.generate_valid_config()
    
    print("=" * 70)
    print(f"Starting automated training run: {run_id}")
    print(f"Generated configuration:")
    for key, value in new_config.items():
        print(f"  {key}: {value}")
    print("=" * 70)
    
    base_config_path = get_absolute_path("config/ppo/3DBall.yaml")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_config_path = temp_file.name
    
    training_start_time = datetime.now()
    
    try:
        ConfigGenerator.update_yaml_config(base_config_path, new_config, temp_config_path)
        
        run_mlagents(run_id, temp_config_path)
        
        training_end_time = datetime.now()
        training_duration = (training_end_time - training_start_time).total_seconds()
        
        data = get_all_data(run_id, temp_config_path)
        data['convergence_detected'] = True
        data['training_duration_seconds'] = int(training_duration)
        data['config_hash'] = str(hash(frozenset(new_config.items())))
        
        for key, value in new_config.items():
            data[key] = value
        
        upsert_training_table(data)
        
        print("=" * 70)
        print(f"Successfully completed run: {run_id}")
        print(f"Training duration: {training_duration:.2f} seconds")
        print(f"Final performance: {data.get('final_10_percent_reward_mean', 'N/A')}")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error in run {run_id}: {e}")
        try:
            partial_data = {
                'run_id': run_id,
                'error_occurred': True,
                'error_message': str(e),
                'training_duration_seconds': int((datetime.now() - training_start_time).total_seconds())
            }
            for key, value in new_config.items():
                partial_data[key] = value
            upsert_training_table(partial_data)
        except Exception as store_error:
            print(f"Failed to store error data: {store_error}")
        raise
    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)

def continuous_training_mode(runs_per_hour: int = 2):
    """Run training continuously with specified frequency"""
    print(f"Starting continuous training mode: {runs_per_hour} runs per hour")
    
    while True:
        try:
            main()
            delay_minutes = 60 / runs_per_hour
            jitter = random.uniform(-5, 5)
            wait_time = max(5, delay_minutes + jitter)
            
            print(f"Waiting {wait_time:.1f} minutes before next run...")
            time.sleep(wait_time * 60)
            
        except KeyboardInterrupt:
            print("\nContinuous training stopped by user")
            break
        except Exception as e:
            print(f"Error in continuous training: {e}")
            print("Waiting 10 minutes before retry...")
            time.sleep(600)

if __name__ == "__main__":
    import sys
    import random
    
    if len(sys.argv) > 1 and sys.argv[1] == "continuous":
        runs_per_hour = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        continuous_training_mode(runs_per_hour)
    else:
        main()