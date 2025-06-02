import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define the base path for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, '01_data_processed')

# Ensure the output directory exists
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)
    print(f"Created directory: {PROCESSED_DATA_DIR}")


def split_data_into_halves(input_df_cleaned):
    """
    Splits the input CLEANED DataFrame into two halves.
    One half retains the 'class' column.
    The other half has its 'class' column removed (features only),
    and its original 'class' labels are saved separately.
    Saves these three DataFrames to CSV files.
    """
    if not isinstance(input_df_cleaned, pd.DataFrame):
        print("Error in split_data_into_halves: input_df_cleaned is not a DataFrame.")
        return None, None, None

    output_filename_h1_with_class = os.path.join(PROCESSED_DATA_DIR, 'mushrooms_h1_with_class.csv')
    output_filename_h2_no_class = os.path.join(PROCESSED_DATA_DIR, 'mushrooms_h2_no_class_features_only.csv')
    output_filename_h2_true_class = os.path.join(PROCESSED_DATA_DIR, 'mushrooms_h2_true_class_only.csv')

    try:
        df_to_split = input_df_cleaned.copy()

        df_h1, df_h2 = train_test_split(
            df_to_split,
            test_size=0.5,
            random_state=42,
            stratify=df_to_split['class']
        )

        df_h2_features_only = df_h2.drop('class', axis=1)
        df_h2_class_original = df_h2['class']

        print("Data_split.py: Data splitting complete.")
        print(f"\nShape of Half 1 (with class): {df_h1.shape}")
        print(f"First 5 rows of Half 1:\n{df_h1.head()}")
        df_h1.to_csv(output_filename_h1_with_class, index=False)
        print(f"Half 1 (with class) saved to '{output_filename_h1_with_class}'")

        print(f"\nShape of Half 2 (features only, class 'deleted'): {df_h2_features_only.shape}")
        print(f"First 5 rows of Half 2 (features only):\n{df_h2_features_only.head()}")
        df_h2_features_only.to_csv(output_filename_h2_no_class, index=False)
        print(f"Half 2 (features only) saved to '{output_filename_h2_no_class}'")

        print(f"\nShape of original 'class' column from Half 2: {df_h2_class_original.shape}")
        print(f"First 5 values of original 'class' from Half 2:\n{df_h2_class_original.head()}")
        df_h2_class_original.to_frame().to_csv(output_filename_h2_true_class, index=False)
        print(f"True 'class' values for Half 2 saved to '{output_filename_h2_true_class}'")

        return df_h1, df_h2_features_only, df_h2_class_original

    except KeyError as e:
        print(f"KeyError during splitting in Data_split.py: {e}. Ensure 'class' column exists in input DataFrame.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred during data splitting in Data_split.py: {e}")
        return None, None, None

# This part is for direct execution of this script (optional, usually called from Main.py)
if __name__ == "__main__":
    # For direct testing, you'd need a cleaned DataFrame.
    # This assumes Data_cleaning.py might have been run and saved a file,
    # or you have a df_cleaned variable available.
    # For simplicity, let's assume you'd load the cleaned data if running standalone.
    try:
        # Example of how you might load cleaned data if running this standalone
        # This path would need to be adjusted based on where the cleaned file is.
        df_cleaned_for_split = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'mushrooms_cleaned.csv'))
        print("Data_split.py: Splitting cleaned mushroom data...")
        split_data_into_halves(df_cleaned_for_split)
    except FileNotFoundError:
        print("Data_split.py: mushrooms_cleaned.csv not found for standalone test. Run Main.py first.")
    except Exception as e:
        print(f"Data_split.py: Error in standalone execution: {e}")

# --- END OF FILE Data_split.py ---