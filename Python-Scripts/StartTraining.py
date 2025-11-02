import os
import sys
from SupabaseAccess import upsert_training_table,check_run_id_exists
from ConsolidateData import get_all_data
from RunMlagents import run_mlagents
from FindAbsPath import get_absolute_path


#######  Temporary hardcoded ######

config_abs_path = get_absolute_path("config")
yaml_abs_path = os.path.join(config_abs_path, "ppo", "3DBall.yaml")
#This will have value: 
#Your Actual Path In Local Machine\ml-agents-group12\config\ppo\3DBall.yaml
########



def main (run_id:str):
    if not check_run_id_exists(run_id):
        try:
            run_mlagents(run_id,yaml_abs_path)
            data = get_all_data(run_id,yaml_abs_path)
            print(data)
            upsert_training_table(data)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
       
    else:
        run_id = "3DBallRunDebug4"  #Default for debugging
    main(run_id)
