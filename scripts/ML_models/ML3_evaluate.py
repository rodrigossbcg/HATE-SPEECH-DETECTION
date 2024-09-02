import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Directory containing study files
studies_dir = "results/ML/finetune"
plots = False

# Import models
models = {
    'LREG': LogisticRegression,
    'LGBM': LGBMClassifier,
    'XGBoost': XGBClassifier,
}

# List all study files in the directory
study_files = [f for f in os.listdir(studies_dir) if f.endswith('.pkl')]

# Loop through each study file
for study_file in study_files:
    study_path = os.path.join(studies_dir, study_file)
    try:
        study = joblib.load(study_path)
        print(f"Loaded study: {study_file}")
    except FileNotFoundError:
        print(f"Study {study_file} not found.")
        continue

    model_name = study_file.split("_")[0]
    encoding = study_file.split("_")[1]
    size = study_file.split("_")[2]
    
    # Load the train data
    train = pd.read_csv(f"data/vectors/{encoding}/train_{size}.csv")
    train_X = train.drop(columns=["label"])
    train_y = train["label"]

    # Load test data
    test = pd.read_csv(f"data/vectors/{encoding}/test_{size}.csv")
    test_X = test.drop(columns=["label"])
    test_y = test["label"]

    # Load the model according to the model name and params
    best_params = study.best_trial.params
    model = models[model_name]()
    model.set_params(**best_params)

    # Fit and evaluate model
    model.fit(train_X, train_y)
    test_pred = model.predict(test_X)

    # Calculate the metrics
    accuracy = accuracy_score(test_y, test_pred)
    recall = recall_score(test_y, test_pred)
    precision = precision_score(test_y, test_pred)
    f1 = f1_score(test_y, test_pred)

    # Save the metrics in csv
    metrics = pd.DataFrame({
        "model": [model_name],
        "encoding": [encoding],
        "size": [size],
        "accuracy": [accuracy],
        "recall": [recall],
        "precision": [precision],
        "f1": [f1]
    })

    metrics.to_csv("results/ML/csv/fine_tune.csv", mode='a', header=False, index=False)


