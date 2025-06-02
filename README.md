# 🍄 Mushroom Classification Project

This project aims to classify mushrooms as **edible** or **poisonous** based on their physical characteristics, utilizing the classic Mushroom Data Set from the UCI Machine Learning Repository. The project demonstrates a comprehensive machine learning workflow, including data loading, cleaning, exploratory data analysis (EDA), data splitting, model training, hyperparameter tuning, evaluation, and final prediction on a held-out set.

---

## 📊 Dataset

*   **Source:** [UCI Mushroom Data Set](https://archive.ics.uci.edu/ml/datasets/Mushroom)
*   **Description:** This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family. Each species is identified as definitely edible or definitely poisonous.
*   **Features:** 22 categorical features and 1 target class.
*   **Samples:** 8,124 instances.
*   **Original File:** The raw `mushrooms.csv` file should be placed in the `00_data_raw/` directory.

---

## 📁 Project Structure
├── 00_data_raw/ # Raw mushroom dataset

├── 01_data_processed/ # Cleaned and processed data

├── 04_reports_and_results/ # Model reports and evaluation results

├── mushrooms_classifications.py # Main pipeline script

├── Data_load.py # Raw data loading

├── Data_cleaning.py # Cleaning and preprocessing

├── Data_split.py # Train/test split

├── EDA.py # Exploratory data analysis

├── Preprocessing_training_prediction.py # Model training and prediction


---

## 🛠️ Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Czerw0/muschrooms-classification.git
    cd muschrooms-classification
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Key libraries include: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`


5.  **Place Dataset:**
    *   Download the `agaricus-lepiota.data` file from the [UCI Mushroom Data Set page](https://archive.ics.uci.edu/ml/datasets/Mushroom).
    *   Rename it to `mushrooms.csv`.
    *   Ensure it has a header row (e.g., `class,cap-shape,cap-surface,...`).
    *   Place `mushrooms.csv` into the `00_data_raw/` directory within the project.

---

## 🚀 Running the Project

The primary way to run the entire pipeline is by executing the main orchestration script:

```bash
python mushrooms_classifications.py
