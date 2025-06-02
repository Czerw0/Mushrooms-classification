# --- START OF FILE Data_load.py ---
import pandas as pd
import os

# Define the base path for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
RAW_DATA_DIR = os.path.join(BASE_DIR, '00_data_raw')
INPUT_CSV_PATH = os.path.join(RAW_DATA_DIR, 'mushrooms.csv')

df_original = None
try:
    df_original = pd.read_csv(INPUT_CSV_PATH)
    print(f"Data_load.py: Successfully loaded {INPUT_CSV_PATH}")
except FileNotFoundError:
    print(f"Error in Data_load.py: '{INPUT_CSV_PATH}' not found. Please ensure the file exists in the '00_data_raw' subfolder.")
except Exception as e:
    print(f"Error loading data in Data_load.py: {e}")

