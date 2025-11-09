import subprocess
import sys
import os
from FindAbsPath import get_absolute_path, get_project_root
import signal
import time
import pandas as pd
import numpy as np
from threading import Thread

"""
ML-Agents Training Runner with Early Stopping

This module provides functionality to execute ML-Agents training runs with automatic
resume capability, cross-platform support, and convergence-based early stopping.
"""

def _check_convergence(run_id: str, window: int = 50, threshold: float = 0.02) -> bool:
    """Check if training has converged using reward stability"""
    try:
        results_path = get_absolute_path("results")
        csv_path = os.path.join(results_path, run_id, f"{run_id}.csv")
        
        if not os.path.exists(csv_path):
            return False
            
        df = pd.read_csv(csv_path)
        if len(df) < window * 2:
            return False
            
        rewards = df['Total_Reward'].values
        recent_rewards = rewards[-window:]
        previous_rewards = rewards[-(window*2):-window]
        
        recent_mean = np.mean(recent_rewards)
        previous_mean = np.mean(previous_rewards)
        
        improvement = (recent_mean - previous_mean) / (abs(previous_mean) + 1e-8)
        return abs(improvement) < threshold
        
    except Exception:
        return False

def _monitor_convergence(run_id: str, process, check_interval: int = 30):
    """Monitor training for convergence and stop when detected"""
    max_checks = 240
    checks = 0
    
    while checks < max_checks and process.poll() is None:
        time.sleep(check_interval)
        checks += 1
        
        try:
            if _check_convergence(run_id):
                print(f"Convergence detected for {run_id}. Stopping training...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
        except Exception as e:
            print(f"Convergence monitoring error: {e}")

def run_mlagents(run_id: str, yaml_abs_path: str):
    """
    This function runs the mlagents-learn command with proper configuration for
    either starting a new training run or resuming an existing one. It automatically
    detects if a run with the given ID already exists and sets the appropriate flags.
    Supports both Windows and Unix-like systems (Linux/Mac).
    
    Args:
        run_id (str): Unique identifier for the training run
        yaml_abs_path (str): Absolute path to the YAML configuration file
    """
    try:
        run_id_exists = _check_run_id_exists_in_results_dir(run_id)     
        resume_flag = "--resume" if run_id_exists else "--force"
        project_root = get_project_root()
        
        if sys.platform == "win32":
            cmd = f"conda activate mlagents && mlagents-learn \"{yaml_abs_path}\" --run-id={run_id} {resume_flag}" 
        else:
            cmd = f"conda run -n mlagents mlagents-learn \"{yaml_abs_path}\" --run-id={run_id} {resume_flag}"
        
        print(f"Executing command: {cmd}")
        process = subprocess.Popen(cmd, shell=True, cwd=project_root)
        
        if not run_id_exists:
            monitor_thread = Thread(target=_monitor_convergence, args=(run_id, process, 30))
            monitor_thread.daemon = True
            monitor_thread.start()

        process.wait()

        if process.returncode == 0:
            print("_" * 70)
            print("\n[INFO] Training completed!\n")
            print("_" * 70)
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

def _check_run_id_exists_in_results_dir(run_id: str):
    results_abs_path = get_absolute_path("results")
    run_path = os.path.join(results_abs_path, run_id)

    if os.path.exists(run_path) and os.path.isdir(run_path):
        return True
    else: 
        return False