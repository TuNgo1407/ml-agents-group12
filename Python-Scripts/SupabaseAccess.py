import os
from supabase import create_client
from postgrest.exceptions import APIError
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd




"""
This module provides functionality to interact with a Supabase database for storing
and retrieving ML training run information. It handles run_id validation and data upsert operations.
"""

current_script_dir = Path(__file__).parent
env_path = current_script_dir / ".env"
load_dotenv(env_path)


"""
    Check if a given run_id already exists in the Supabase training table.
    
    This function queries the Supabase database to verify whether a training run
    with the specified ID already exists, preventing duplicate run IDs in the system.
    
    Args:
        run_id_need_to_check (str): The run_id to check for existence in the database
    
    Returns:
        bool: True if the run_id exists in the database, False if it doesn't exist
              or if an error occurred during the check
    """
def check_run_id_exists(run_id_need_to_check:str) -> bool:
	print("[INFO] Checking if run_id exists in database...")
	try: 
		training_table = os.environ.get("TRAINING_TABLE")
		url = os.environ.get("SUPABASE_URL")
		key = os.environ.get("SUPABASE_KEY")
		run_id_primary_key = 'run_id'
		supabase = create_client(url, key)

		response = supabase.table(training_table)\
			.select(run_id_primary_key)\
			.eq(run_id_primary_key,run_id_need_to_check)\
			.execute()

		exsits = len(response.data) > 0
		if exsits:
			print(f"[Error] run_id: {run_id_need_to_check} already exists, please try another run_id")

		return exsits
	
	except APIError as e:
		print(f"Supabase API error: {e}")
		return False

	except Exception as e:
		print(f"Unexpected error: {e}")
		return False


"""
    Insert or update training data in the Supabase training table.
    
    This function performs an upsert operation (insert or update) on the training table
    using the provided data dictionary. If a record with the same run_id already exists,
    it will be updated; otherwise, a new record will be inserted.
    
    Args:
        data_dict (dict): A dictionary containing the training data to upsert.
                         Must include a 'run_id' field for the conflict resolution.
    
"""

def upsert_training_table (data_dict : dict):
	
	try:
		training_table = os.environ.get("TRAINING_TABLE")
		url = os.environ.get("SUPABASE_URL")
		key = os.environ.get("SUPABASE_KEY")

		if not url or not key:
			raise ValueError ("Missing Supabase URL or Key")
		supabase = create_client(url,key)

		if not data_dict:
			raise ValueError ("Data dict is empty")
		
		# records = df.to_dict('records')
		response = supabase.table(training_table).upsert(data_dict, on_conflict="run_id").execute()

		print("_" * 70)
		print("\n[INFO] Updated record to database successfully!\n")
		print("_" * 70)
		print(f"\nRecord:\n")

	
		if response.data and len(response.data) > 0:
			record = response.data[0]  
			for key, value in record.items():
				print(f"{key} : {value}")
		else:
			print("No data returned in response")
		print("_" * 70)
	
	except ValueError as e:
		print (f"Error: {e}")
		raise
	except APIError as e:
		print (f"Supabase API Error")
		raise
