from FindAbsPath import get_absolute_path
import pandas as pd
import numpy as np
import os

def get_epi_training_data (run_id:str):
    try: 
        path_to_result_dir = get_absolute_path("results")

        if not os.path.exists(path_to_result_dir):
            raise FileNotFoundError(f"Results folder not found: {path_to_result_dir}")

        path_to_ep_csv = os.path.join(path_to_result_dir, run_id,f"{run_id}.csv")

        if not os.path.exists(path_to_ep_csv):
            raise FileNotFoundError(f"Episode CSV not found: {path_to_ep_csv}")

        df = pd.read_csv(path_to_ep_csv)


        #Just in case we need more information, and for later use. 
        # df = df.reset_index(drop=True)
        # df['Actual_Episode'] = df.index + 1
        # df['Training Session'] = (df['Episode']==1).cumsum()

        rewards = df['Total_Reward'].tolist()
        total_episodes = len(df)
        if total_episodes < 1:
            raise ValueError(f"Insufficient episodes: {total_episodes}")

        final_10_percent_count = max(1, int(total_episodes * 0.1))  
        final_rewards = rewards[-final_10_percent_count:]
        final_10_percent_reward_mean = np.mean(final_rewards)


        all_rewards_std = np.std(rewards)

        epi_dict = {"run_id": run_id
                 ,"final_10_percent_reward_mean":final_10_percent_reward_mean
                 ,"all_rewards_std":all_rewards_std
                 }
        return epi_dict
    except FileNotFoundError as e:
        raise 
    except ValueError as e:
        raise 


