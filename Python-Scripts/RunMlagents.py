import subprocess
import sys
import os
from FindAbsPath import get_absolute_path,get_project_root

"""
ML-Agents Training Runner

This module provides functionality to execute ML-Agents training runs with automatic
resume capability and cross-platform support.
"""



"""
    This function runs the mlagents-learn command with proper configuration for
    either starting a new training run or resuming an existing one. It automatically
    detects if a run with the given ID already exists and sets the appropriate flags.
    Supports both Windows and Unix-like systems (Linux/Mac).
    
    Args:
        run_id (str): 
        yaml_abs_path (str): Absolute path to the YAML configuration file
    
"""



def run_mlagents(run_id:str, yaml_abs_path:str):
   
    try:
      
        run_id_exists = _check_run_id_exists_in_results_dir(run_id)     
        resume_flag = "--resume" if run_id_exists else "--force"

        project_root  = get_project_root()
          
        if sys.platform == "win32":
            cmd = f"conda activate mlagents && mlagents-learn \"{yaml_abs_path}\" --run-id={run_id} {resume_flag}" 
            process = subprocess.Popen(cmd, shell=True,cwd=project_root)
        else:
        # Linux/Mac
            cmd = f"conda activate mlagents && mlagents-learn \"{yaml_abs_path}\" --run-id={run_id} {resume_flag}" 
            process = subprocess.Popen(cmd, shell=True, executable='/bin/bash',cwd=project_root)
        
        process.wait()

        if process.returncode == 0:
            print("_"*70)
            print("\n[INFO] Training completed!\n")
            print("_"*70)
        else:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        
    except FileNotFoundError as e:
        print(f"Error: Command not found. Check if conda and mlagents-learn are installed {e}")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error: Training failed with exit code {e.returncode}")
        raise
    except Exception as e:
        print(f"Unexpected error during training: {e}")
        raise

    

def _check_run_id_exists_in_results_dir(run_id:str):
    results_abs_path = get_absolute_path("results")
    run_path = os.path.join(results_abs_path, run_id)

    #check if path exists and path path is directory
    if os.path.exists(run_path) and os.path.isdir(run_path):
        return True
    else: return False

# if __name__ == "__main__":
#     run_mlagents("2")