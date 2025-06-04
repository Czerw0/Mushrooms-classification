# 🍄 Mushroom Classification Project

A machine learning pipeline to classify mushrooms as **edible** or **poisonous** using the UCI Mushroom Dataset. This project demonstrates end-to-end data handling — from ingestion and cleaning to modeling, evaluation, and final predictions.

---

## 📊 Dataset Information

*   **Source:** [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom)
*   **Description:** Hypothetical mushroom species described by 22 categorical features, labeled as either **edible** or **poisonous**.
*   **Instances:** 8,124 samples
*   **Features:** 22 categorical attributes
*   **Target:** `class` — edible or poisonous
*   **Usage Note:** Place the dataset file `mushrooms.csv` in the `00_data_raw/` directory.

---

## 🧪 Machine Learning Workflow

*   **Data Loading:**
    *   Load raw dataset from `00_data_raw/` using `Data_load.py`.

*   **Data Cleaning & Transformation:**
    *   Use `Data_cleaning.py` to:
        * Convert encoded characters (e.g., `'e'` → `'edible'`, `'p'` → `'poisonous'`).
        * Save cleaned data to `01_data_processed/`.

*   **Exploratory Data Analysis (EDA):**
    *   Generate distribution plots and class-wise feature visualizations using `EDA.py`.
    *   Visuals saved to `04_reports_and_results/`.

*   **Data Splitting:**
    *   Split cleaned data using `Data_split.py` into:
        * `h1_with_class`: used for model training and validation.
        * `h2_features_only` and `h2_true_class`: held-out set for final predictions.

*   **Model Training & Evaluation:**
    *   Use `Preprocessing_training_prediction.py` to:
        * Perform one-hot encoding and label encoding.
        * Train baseline models (e.g., Decision Tree, Random Forest).
        * Evaluate using validation data (accuracy, confusion matrix, ROC curves).
        * Perform k-fold cross-validation.
        * Use `GridSearchCV` for hyperparameter tuning.
        * (Optional) Save trained model and encoders to `03_models/`.

*   **Final Prediction (on H2 set):**
    *   Predict on the held-out test set (`h2_features_only`) and compare with true labels (`h2_true_class`).
    *   Save final reports, metrics, and predictions to `04_reports_and_results/` and `01_data_processed/`.
---
## Thoughts on Project 
It may seems that the model is overtrained. However this dataset is very simple so it can actually be really accurate

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
## 📈 Outputs & Reports 

After execution, reports and visualizations will be stored in 04_reports_and_results/.

* **EDA Visuals:**
    * Class distribution plots
    * Feature distributions split by class
    * Model Evaluation Metrics:
    * Confusion matrices for each model
* **ROC curves:**
    * Cross-validation score summary
    * Final Evaluation on H2:
    * Confusion matrix and ROC curve
    * Final prediction CSV: mushrooms_h2_final_predictions.csv
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


4.  **Place Dataset:**
    *   Download the `agaricus-lepiota.data` file from the [UCI Mushroom Data Set page](https://archive.ics.uci.edu/ml/datasets/Mushroom).
    *   Rename it to `mushrooms.csv`.
    *   Ensure it has a header row (e.g., `class,cap-shape,cap-surface,...`).
    *   Place `mushrooms.csv` into the `00_data_raw/` directory within the project.

---

## 🚀 Running the Project

The primary way to run the entire pipeline is by executing the main orchestration script:

```bash
python mushrooms_classifications.py
```

---

## ⚠️ Disclaimer
This is a machine learning demo project.
Do NOT use it to classify or consume wild mushrooms.
Misidentifying mushrooms can be fatal — always consult a professional mycologist.
