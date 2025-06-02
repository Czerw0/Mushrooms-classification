import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, '01_data_processed')
REPORTS_DIR = os.path.join(BASE_DIR, '04_reports_and_results') # For saving plots
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    print(f"Created directory: {REPORTS_DIR}")

def perform_eda(df_for_eda):
    if not isinstance(df_for_eda, pd.DataFrame):
        print("Error in perform_eda: df_for_eda is not a DataFrame.")
        return

    print("\nEDA.py: Performing EDA...")
    print("Dataset Information:")
    df_for_eda.info()

    print("\nFirst 5 rows of the dataset:")
    print(df_for_eda.head())

    print("\nSummary Statistics:")
    print(df_for_eda.describe(include='all').T)

    print("\nMissing Values:")
    print(df_for_eda.isnull().sum())

    print("\nClass Distribution:")
    print(df_for_eda['class'].value_counts())

    # Visualize the distribution of the target variable
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_for_eda, x='class', palette='Set2')
    plt.title('Distribution of Target Variable (Class)')
    plt.savefig(os.path.join(REPORTS_DIR, 'class_distribution.png'))
    plt.close() 
    print("EDA.py: Saved class distribution plot.")


    # Visualize the distribution of categorical features
    print("EDA.py: Generating and saving feature distribution plots...")
    categorical_features = df_for_eda.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        if feature == 'class': # Already plotted
            continue
        plt.figure(figsize=(12, 7)) # Adjusted size
        try:
            sns.countplot(data=df_for_eda, x=feature, hue='class', palette='Set2', order = df_for_eda[feature].value_counts().index)
            plt.title(f'Distribution of {feature} by Class')
            plt.xticks(rotation=45, ha="right") # ha="right" for better alignment
            plt.legend(title='Class')
            plt.tight_layout() # Adjust layout to prevent labels from overlapping
            plt.savefig(os.path.join(REPORTS_DIR, f'distribution_{feature}_by_class.png'))
            plt.close() # Close each plot after saving
        except Exception as e:
            print(f"Could not plot {feature}: {e}")
    print("EDA.py: Feature distribution plots saved.")
    print("EDA.py: EDA complete.")

# This part is for direct execution of this script
if __name__ == "__main__":
    '''
    The charts are directly saved to the '04_reports_and_results' subfolder.
    '''
    try:
        df_cleaned = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'mushrooms_cleaned.csv'))
        perform_eda(df_cleaned)
    except FileNotFoundError:
        print("EDA.py: 'mushrooms_cleaned.csv' not found in 01_data_processed. Run Main.py to generate it.")
    except Exception as e:
        print(f"EDA.py: Error in standalone execution: {e}")
