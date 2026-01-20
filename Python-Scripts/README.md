# Training ML-Agents Guidelines

## Building your Enviorment in Unity
First, Make sure you have installed:
- python 3.10.11 (in your env)
- mlagent 1.1.0
- Unity Editor 2023.2.12f1

And completed the [Installation Guide](https://github.com/TuNgo1407/ml-agents-group12/blob/develop/docs/Readme.md) beforehand.

1. Navigate to the 3DBall scene in the Unity Editor (`Assets/ML-Agents/Examples/3DBall/Scenes`)
2. Double click the 3DBall scene
3. At the top left, under files click `Build Settings...`
4. Click `Add Open Scenes` and check `Development Build`
5. Make sure the `Target Platform` and `Architecture` correspond with your machine
6. Click build and create a folder within `ml-agents-group12` to contain the build 
7. Go to `Python-Scripts/RunMlagents.py` line 82. Check if the path corresponds with your build 

### Analysis Subfolder

The `Analysis` subfolder contains tools for data analysis and visualization. It includes:
- **Analysis.ipynb**: A Jupyter notebook for running analyses on the training data. It utilizes the `PullData.py` script to fetch data from Supabase and visualize results using libraries like Pandas, Seaborn, and Matplotlib.
- **PullData.py**: A script that connects to Supabase to retrieve training data. Ensure that your environment variables for Supabase are correctly set in the `.env` file to enable data fetching.

### Run ML-Agents Training

1. On your terminal, navigate (cd) to directory `ml-agents-group12/Python-Scripts`
    ```bash
    cd ml-agents-group12/Python-Scripts
    ```
2. Activate `mlagents env` and disable saving unnecessary files simultaneously by this command: 
    ```bash
    conda activate mlagents && python disable_model_saving.py
    ```
    You will see this result:
    ```
    Successfully disabled model saving and ONNX export in ML-Agents
    ```
3. Train a single run with the first command or do multiple with the second:
    ```bash
    python StartTraining.py
    ```
    ```bash 
    python ContinuousTraining.py
    ```
4. **WAIT** until you see this output
    <details>
    <summary>Click to view exmaple output</summary>

    ```
    ======================================================================
    Starting automated training run: 3DBall_20251111_163319_mw96qq (run_id of current training session)
    Generated configuration:
      learning_rate: 0.0001
      batch_size: 256
      buffer_size: 4096
      hidden_units: 128
      num_layers: 1
      time_horizon: 64
      beta: 0.0003
      gamma: 0.995
      lambd: 0.95
      max_steps: 1000000
    ======================================================================
    Executing command: conda activate mlagents && mlagents-learn "path to temporary .yaml" --run-id=3DBall_20251111_163319_mw96qq --force

                ┐  ╖
            ╓╖╬│╡  ││╬╖╖
        ╓╖╬│││││┘  ╬│││││╬╖
     ╖╬│││││╬╜        ╙╬│││││╖╖                               ╗╗╗
     ╬╬╬╬╖││╦╖        ╖╬││╗╣╣╣╬      ╟╣╣╬    ╟╣╣╣             ╜╜╜  ╟╣╣
     ╬╬╬╬╬╬╬╬╖│╬╖╖╓╬╪│╓╣╣╣╣╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╒╣╣╖╗╣╣╣╗   ╣╣╣ ╣╣╣╣╣╣ ╟╣╣╖   ╣╣╣
     ╬╬╬╬┐  ╙╬╬╬╬│╓╣╣╣╝╜  ╫╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╟╣╣╣╙ ╙╣╣╣  ╣╣╣ ╙╟╣╣╜╙  ╫╣╣  ╟╣╣
     ╬╬╬╬┐     ╙╬╬╣╣      ╫╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╟╣╣╬   ╣╣╣  ╣╣╣  ╟╣╣     ╣╣╣┌╣╣╜
     ╬╬╬╜       ╬╬╣╣      ╙╝╣╣╬      ╙╣╣╣╗╖╓╗╣╣╣╜ ╟╣╣╬   ╣╣╣  ╣╣╣  ╟╣╣╦╓    ╣╣╣╣╣
     ╙   ╓╦╖    ╬╬╣╣   ╓╗╗╖            ╙╝╣╣╣╣╝╜   ╘╝╝╜   ╝╝╝  ╝╝╝   ╙╣╣╣    ╟╣╣╣
       ╩╬╬╬╬╬╬╦╦╬╬╣╣╗╣╣╣╣╣╣╣╝                                             ╫╣╣╣╣
          ╙╬╬╬╬╬╬╬╣╣╣╣╣╣╝╜
              ╙╬╬╬╣╣╣╜
                 ╙

     Version information:
      ml-agents: 1.1.0,
      ml-agents-envs: 1.1.0,
      Communicator API: 1.5.0,
      PyTorch: 2.1.1+cu118
    [INFO] Listening on port 5004. Start training by pressing the Play button in the Unity Editor.
    ```

    </details>

5. After the current training session is completed, you will see this output:
    <details>
    <summary>Click to view example output</summary>

    ```
    ______________________________________________________________________
    
    [INFO] Training completed!
    
    ______________________________________________________________________
    ______________________________________________________________________
    
    [INFO] Updated record to database successfully!
    
    ______________________________________________________________________
    
    Record:
    
    run_id : 3DBall_20251111_163319_mw96qq
    learning_rate : 0.0001
    batch_size : 256
    buffer_size : 4096
    hidden_units : 128
    num_layers : 1
    time_horizon : 64
    beta : 0.0003
    gamma : 0.995
    lambd : 0.95
    final_10_percent_reward_mean : 385.332265306123
    mean_reward : 100.374697247706
    max_reward : 500.02
    min_reward : 4.9
    late_phase_std : 185.239624943914
    all_rewards_std : 157.665983348841
    convergence_detected : True
    convergence_episode : 53
    episodes_after_convergence : 4852
    total_episodes : 4905
    training_duration_seconds : 860
    config_hash : 546699675737269498
    config_path : None
    error_occurred : False
    created_at : 2025-11-11T15:47:44.389273
    updated_at : 2025-11-11T15:47:44.389273
    error_message : None
    max_steps : 1000000
    ______________________________________________________________________
    ======================================================================
    Successfully completed run: 3DBall_20251111_163319_mw96qq
    Training duration: 860.70 seconds
    Final performance: 385.3322653061225
    ======================================================================

    ```


6. If you are still have the `mlagents env` opened, you can go back from `Step 3` to start a new training session. Otherwise please restart from the beginning.

**Warning**: Training must run to completion without interruption. Any interruption requires starting over and no data will be saved to the database.
## Check our outputs on Supabase Database
Access [Supabase Database](https://supabase.com/dashboard/project/wkxpgbzzzsbnbktevgkg/editor/21339) 

**Note 1:** Please **DO NOT** perform any **queries** or **table modification**.

**Note 2:** **AFTER** the training session is completed. You can delete all the files to save space on your machine.


