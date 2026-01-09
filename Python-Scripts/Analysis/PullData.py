from supabase import create_client
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd




current_script_dir = Path(__file__).parent.parent
env_path = current_script_dir / ".env"
load_dotenv(env_path)


def pull_data_from_supabase() -> pd.DataFrame:
    training_table = os.environ.get("TRAINING_TABLE")
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    supabase = create_client(url,key)

    response = supabase.table(training_table) \
    .select("*") \
    .execute()
    data = response.data

    return pd.DataFrame(data)

