import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, '01_data_processed') # Directory for processed data
MODELS_DIR = os.path.join(BASE_DIR, '03_models') # For label encoder
REPORTS_DIR = os.path.join(BASE_DIR, '04_reports_and_results') # For saving confusion matrix plot

if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

# Ensure the processed data directory exists
try:
    df_h1_path = os.path.join(PROCESSED_DATA_DIR, 'mushrooms_h1_with_class.csv')
    df_h1 = pd.read_csv(df_h1_path)
    print(f"Successfully loaded: {df_h1_path}")
except FileNotFoundError:
    print(f"Error: {df_h1_path} not found. Make sure Data_split.py has been run.")
    exit() # Or handle error appropriately
except Exception as e:
    print(f"Error loading {df_h1_path}: {e}")
    exit()



if __name__ == "__main__":
    TARGET_COLUMN = 'class'
    # Data Loading and Preparation
    if TARGET_COLUMN not in df_h1.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in df_h1.")
        print(f"Available columns are: {df_h1.columns.tolist()}")
        exit()

    X_h1 = df_h1.drop(TARGET_COLUMN, axis=1)
    y_h1 = df_h1[TARGET_COLUMN] # Target as strings

    # Encode Features (X_h1) using one-hot encoding
    print("\nEncoding features (X_h1)...")
    X_h1_encoded = pd.get_dummies(X_h1, prefix_sep='_', dummy_na=False)
    print(f"Shape of X_h1_encoded: {X_h1_encoded.shape}")
    print(f"First 5 rows of X_h1_encoded (features):\n{X_h1_encoded.head()}")

    # Encode Target (y_h1_str) to numerical labels (0, 1)
    print("\nEncoding target variable (y_h1)...")
    label_encoder_y = LabelEncoder()
    y_h1_encoded = label_encoder_y.fit_transform(y_h1)
    print(f"Shape of y_h1_encoded: {y_h1_encoded.shape}")

    # X_train, X_test, y_train, y_test Split
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
    X_h1_encoded, y_h1_encoded, test_size=0.2, random_state=42, stratify=y_h1_encoded
    )
    
    #Models Training and Evaluation
    print("--- Training Models ---")
    clf = LogisticRegression(random_state=0, solver='liblinear', max_iter=200).fit(X_train, y_train)
    rfc = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    svc = SVC(random_state=0).fit(X_train, y_train)
    knn = KNeighborsClassifier().fit(X_train, y_train)
    dtr = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    print("Models trained.")

    print("\n--- Making Predictions ---")
    y_pred_clf = clf.predict(X_test)
    y_pred_rfc = rfc.predict(X_test)
    y_pred_svc = svc.predict(X_test)
    y_pred_knn = knn.predict(X_test)
    y_pred_dtr = dtr.predict(X_test) 
    print("Predictions made.")

    print("LogisticRegression")
    print(classification_report(y_test, y_pred_clf, target_names=label_encoder_y.classes_))

    print("\n")
    print("RandomForestClassifier")
    print(classification_report(y_test, y_pred_rfc, target_names=label_encoder_y.classes_))
    print("\n")
    print("SVC")
    print(classification_report(y_test, y_pred_svc, target_names=label_encoder_y.classes_))
    print("\n")
    print("KNeighborsClassifier")
    print(classification_report(y_test, y_pred_knn, target_names=label_encoder_y.classes_))
    print("\n")
    print("Decision Tree")
    print(classification_report(y_test, y_pred_dtr, target_names=label_encoder_y.classes_))


    rfc_cv = RandomForestClassifier()
    svc_cv = SVC()
    knn_cv = KNeighborsClassifier()
    logr_cv = LogisticRegression()
    dtr_cv = DecisionTreeClassifier()

    models = [rfc_cv,svc_cv,knn_cv,logr_cv, dtr_cv]

    for model in models:
        model.fit(X_train, y_train)

        # Confusion Matrix
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=label_encoder_y.classes_)
        disp.ax_.set_title(f"Confusion Matrix for {model.__class__.__name__}")
        plt.savefig(os.path.join(REPORTS_DIR, f"confusion_matrix_{model.__class__.__name__}.png"))
        plt.clf()

        # ROC Curve (only for models that support it)
        if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
            try:
                roc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
                roc_disp.ax_.set_title(f"ROC Curve for {model.__class__.__name__}")
                plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
                plt.savefig(os.path.join(REPORTS_DIR, f"roc_curve_{model.__class__.__name__}.png"))
                plt.clf()
            except Exception as e:
                print(f"Could not plot ROC for {model.__class__.__name__}: {e}")

        # Cross-validation on full training set
        scores = cross_val_score(model, X_train, y_train, cv=10)
        print(f"{model.__class__.__name__}: mean CV accuracy = {scores.mean():.4f}, std = {scores.std():.4f}")


    #Best is random forrest with highest mean and lowest std

    # Grid Search for Hyperparameter Tuning
    grid = {
    #'bootstrap': [True],
    'max_depth': [5,8,10],
    'max_features': ['sqrt', 'log2',None,5],
    #'min_samples_leaf': [3, 4, 5],
    #'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200,500]
    }
    rfc_grid = RandomForestClassifier()
    rfc_grid_cv=GridSearchCV(rfc_grid,grid,cv=10)
    rfc_grid_cv.fit(X_train,y_train)

    print("tuned hpyerparameters :(best parameters) ",rfc_grid_cv.best_params_)
    print("accuracy :",rfc_grid_cv.best_score_)
    
    #Best Model Training
    best_rfc = rfc_grid_cv.best_estimator_
    best_rfc.fit(X_train, y_train)
    print("Best Random Forest Classifier trained with tuned hyperparameters.")

    # Half2 Encoding and Predictions based on Best Model
    try:
        df_h2_path = os.path.join(PROCESSED_DATA_DIR, 'mushrooms_h2_no_class_features_only.csv')
        df_h2 = pd.read_csv(df_h2_path)
        print(f"Successfully loaded: {df_h2_path}")
    except FileNotFoundError:
        print(f"Error: {df_h2_path} not found. Make sure Data_split.py has been run.")
        exit()
    except Exception as e:
        print(f"Error loading {df_h2_path}: {e}")
        exit()

    print("\nEncoding features (H2)...")
    X_h2_encoded = pd.get_dummies(df_h2, prefix_sep='_', dummy_na=False)

    # Align columns with training set (X_h1_encoded)
    X_h2_encoded = X_h2_encoded.reindex(columns=X_h1_encoded.columns, fill_value=0)
    print(f"Shape of X_h2_encoded: {X_h2_encoded.shape}")
    print(f"First 5 rows of X_h2_encoded:\n{X_h2_encoded.head()}")

    #Make Predictions on H2 (Unlabeled Data)
    print("\nMaking predictions on unlabeled data (H2) with best Random Forest model...")
    h2_predictions = best_rfc.predict(X_h2_encoded)
    h2_pred_labels = label_encoder_y.inverse_transform(h2_predictions)

    print(f"Predictions for H2: {h2_pred_labels[:5]}...")  #
    # Join predictions with original H2 dataset and decode labels
    h2_results = df_h2.copy()
    h2_results['predicted_class_encoded'] = h2_predictions
    h2_results['predicted_class'] = h2_pred_labels

    #Exporting Final Predictions
    output_path = os.path.join(REPORTS_DIR, 'Final_predictions.csv')
    h2_results.to_csv(output_path, index=False)
    print(f"Final predictions saved to: {output_path}")

