from Data_load import df_original
import pandas as pd
from sklearn.model_selection import train_test_split


def split():

    input_filename = df_original.copy()

    output_filename_h1_with_class = 'mushrooms_h1_with_class.csv'
    output_filename_h2_no_class = 'mushrooms_h2_no_class_.csv'
    # Optional: to save the true classes of half2 for later verification
    output_filename_h2_true_class = 'mushrooms_h2_true_class_only.csv'


    try:
        # 1. Load the Data
        df_original = pd.read_csv(input_filename)

        # 2. Split the DataFrame into two halves (50/50)
        # We use stratify on the 'class' column to ensure both halves have a similar
        # proportion of poisonous/edible mushrooms as the original dataset.
        df_h1, df_h2 = train_test_split(
            df_original,
            test_size=0.5,         # 50% of the data goes to df_half2
            random_state=42,       # For reproducible splits
            stratify=df_original['class']   # Stratify by the 'class' column
        )

        # 3. For the second half (df_half2), separate its features and its 'class' column
        # Then, we will "delete" the class column from the features part.
        df_h2_features_only = df_h2.drop('class', axis=1)
        df_h2_class_original = df_h2['class'] # This is the "deleted" class column

        # At this point:
        # - df_h1: Contains 50% of the original data, including the 'class' column.
        # - df_h2_features_only: Contains the features of the other 50%,
        #                           with the 'class' column removed.
        # - df_h2_class_original: Contains only the 'class' column for the second half,
        #                            which we've "hidden" for later classification/verification.

        print("Data splitting complete.")
        print(f"\nShape of Half 1 (with class): {df_h1.shape}")
        print(f"First 5 rows of Half 1:\n{df_h1.head()}")

        print(f"\nShape of Half 2 (features only, class 'deleted'): {df_h2_features_only.shape}")
        print(f"First 5 rows of Half 2 (features only):\n{df_h2_class_original.head()}")

        print(f"\nShape of original 'class' column from Half 2: {df_h2_class_original.shape}")
        print(f"First 5 values of original 'class' from Half 2:\n{df_h2_class_original.head()}")


        # 4. Optional: Save the resulting DataFrames to new CSV files
        df_h1.to_csv(output_filename_h1_with_class, index=False)
        df_h2_features_only.to_csv(output_filename_h2_no_class, index=False)
        df_h2_class_original.to_csv(output_filename_h2_true_class, index=False, header=['class']) # Save as a Series with header

        print(f"\nHalf 1 (with class) saved to '{output_filename_h1_with_class}'")
        print(f"Half 2 (features only) saved to '{output_filename_h2_no_class}'")
        print(f"True 'class' values for Half 2 saved to '{output_filename_h2_true_class}'")


    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
    except Exception as e:
        print(f"An error occurred during data splitting: {e}")

