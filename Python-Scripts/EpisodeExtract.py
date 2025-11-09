from FindAbsPath import get_absolute_path
import pandas as pd
import numpy as np
import os

def get_epi_training_data(run_id: str):
    """
    Extract training performance metrics from episode CSV data
    with enhanced convergence-aware analysis
    """
    try: 
        path_to_result_dir = get_absolute_path("results")
        path_to_ep_csv = os.path.join(path_to_result_dir, run_id, f"{run_id}.csv")
        path_to_behavior_csv = os.path.join(path_to_result_dir, run_id, "3DBall.csv")
        
        if os.path.exists(path_to_ep_csv):
            df = pd.read_csv(path_to_ep_csv)
        elif os.path.exists(path_to_behavior_csv):
            df = pd.read_csv(path_to_behavior_csv)
        else:
            raise FileNotFoundError(f"Episode CSV not found at either {path_to_ep_csv} or {path_to_behavior_csv}")
        total_episodes = len(df)
        
        if total_episodes < 10:
            raise ValueError(f"Insufficient episodes: {total_episodes}")

        rewards = df['Total_Reward'].tolist()
        
        final_10_percent_count = max(1, int(total_episodes * 0.1))  
        final_rewards = rewards[-final_10_percent_count:]
        final_10_percent_reward_mean = np.mean(final_rewards)
        
        late_phase_count = max(1, int(total_episodes * 0.5))
        late_phase_rewards = rewards[-late_phase_count:]
        late_phase_std = np.std(late_phase_rewards)
        
        all_rewards_std = np.std(rewards)
        max_reward = np.max(rewards)
        min_reward = np.min(rewards)
        
        convergence_metrics = _calculate_convergence_metrics(rewards)
        
        epi_dict = {
            "run_id": run_id,
            "final_10_percent_reward_mean": final_10_percent_reward_mean,
            "late_phase_std": late_phase_std,
            "all_rewards_std": all_rewards_std,
            "total_episodes": total_episodes,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "mean_reward": np.mean(rewards),
            **convergence_metrics
        }
        return epi_dict
        
    except FileNotFoundError as e:
        raise 
    except ValueError as e:
        raise 
    except Exception as e:
        print(f"Error processing training data for {run_id}: {e}")
        raise

def _calculate_convergence_metrics(rewards, window=50):
    """Calculate convergence-related metrics"""
    if len(rewards) < window * 2:
        return {
            "convergence_episode": None,
            "episodes_after_convergence": 0,
            "convergence_detected": False
        }
    
    for i in range(window, len(rewards) - window):
        prev_window = rewards[i-window:i]
        current_window = rewards[i:i+window]
        
        prev_mean = np.mean(prev_window)
        current_mean = np.mean(current_window)
        
        improvement_ratio = abs(current_mean - prev_mean) / (abs(prev_mean) + 1e-8)
        
        if improvement_ratio < 0.02:
            return {
                "convergence_episode": i,
                "episodes_after_convergence": len(rewards) - i,
                "convergence_detected": True
            }
    
    return {
        "convergence_episode": None,
        "episodes_after_convergence": 0,
        "convergence_detected": False
    }