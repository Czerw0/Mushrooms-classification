# --- START OF FILE Main.py ---
import pandas as pd
import os

# Import functions from your other modules
from Data_load import df_original # This will be None if loading failed
from Data_cleaning import clean_mushroom_data_descriptive
from Data_split import split_data_into_halves
from EDA import perform_eda # Optional: if you want EDA as part of the main flow

# Define base and processed data directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, '01_data_processed')

# Ensure the output directory exists
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)
    print(f"Main.py: Created directory: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    print("Main.py: Starting mushroom data processing pipeline...")

    if df_original is None:
        print("Main.py: df_original not loaded from Data_load.py. Exiting.")
    else:
        # 1. Clean the dataset (descriptive values)
        print("\nMain.py: Cleaning mushroom data...")
        df_cleaned = clean_mushroom_data_descriptive(df_original)

        if df_cleaned is not None:
            # Save the cleaned dataset
            output_filename_cleaned = os.path.join(PROCESSED_DATA_DIR, 'mushrooms_cleaned.csv')
            df_cleaned.to_csv(output_filename_cleaned, index=False)
            print(f"Main.py: Cleaned data (descriptive) saved to '{output_filename_cleaned}'")
            print(f"First 5 rows of cleaned data:\n{df_cleaned.head()}")

            # 2. Perform EDA on the cleaned data (optional step here)
            perform_eda(df_cleaned) # Call EDA function with the cleaned data

            # 3. Split the cleaned data
            print("\nMain.py: Splitting the cleaned data...")
            # The split_data_into_halves function now handles saving its outputs
            df_h1, df_h2_features, df_h2_true_class = split_data_into_halves(df_cleaned)

            if df_h1 is not None:
                print("Main.py: Data splitting successful.")
                # Further steps (like preprocessing for ML, training, etc.) would follow here
                # using df_h1, df_h2_features, and df_h2_true_class.
            else:
                print("Main.py: Data splitting failed.")
        else:
            print("Main.py: Data cleaning failed. Skipping subsequent steps.")

    print("\nMain.py: Pipeline finished.")

# --- END OF FILE Main.py ---