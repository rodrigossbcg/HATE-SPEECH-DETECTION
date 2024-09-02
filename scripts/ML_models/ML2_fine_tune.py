import pandas as pd
import os

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import joblib

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner


# Import models
models = {
    'LREG': LogisticRegression,
    'LGBM': LGBMClassifier,
    'XGBoost': XGBClassifier,
}


def objective(trial, X_train, y_train, X_test, y_test, model_name):

    if model_name == 'LREG':

        # Hyperparameter space
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])
        solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'newton-cg', 'saga'])

        # Ensure compatibility of penalty and solver
        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
            raise optuna.TrialPruned(f"Solver {solver} supports only 'l2' or 'none' penalties, got l1 penalty.")
        if penalty == 'elasticnet' and solver != 'saga':
            raise optuna.TrialPruned("Elasticnet penalty is only supported by the 'saga' solver.")
        if penalty is None and solver not in ['lbfgs', 'newton-cg', 'sag', 'saga']:
            raise optuna.TrialPruned(f"Solver {solver} does not support 'none' penalty.")
        if solver == 'liblinear' and penalty == 'elasticnet':
            raise optuna.TrialPruned("Liblinear solver does not support 'elasticnet' penalty.")

        params = {
            'penalty': penalty,
            'solver': solver,
            'dual': False,
            'tol': trial.suggest_loguniform('tol', 1e-5, 1e-1),
            'C': trial.suggest_loguniform('C', 1e-4, 1e2),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'intercept_scaling': 1,
            'max_iter': trial.suggest_int('max_iter', 50, 500),
            'warm_start': trial.suggest_categorical('warm_start', [True, False]),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0) if penalty == 'elasticnet' else None
        }

        # Model definition
        model = LogisticRegression(**params)


    elif model_name == 'LGBM':

        # Hyperparameter space
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'num_iterations': trial.suggest_int('num_iterations', 100, 2000),  # Wider range for iterations
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),  # Extended lower bound
            'num_leaves': trial.suggest_int('num_leaves', 2, 512, log=True),  # More options for leaves
            'max_depth': trial.suggest_int('max_depth', -1, 30),  # Extended range for depth
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 200),  # Adjusted for larger datasets
            'max_bin': trial.suggest_int('max_bin', 63, 255),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 10.0),
            'random_state': 1,
            'verbose': -1
        }

        # Model definition
        model = LGBMClassifier(**params)


    else:

        # Hyperparameter space
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),  # Extended range for more iterations
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),  # This range is good
            'max_depth': trial.suggest_int('max_depth', 1, 30),  # Extended range for more depth
            'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),  # This range is good
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),  # This range is good
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # This range is good
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),  # This range is good
            'lambda': trial.suggest_float('lambda', 0.0, 10.0),  # This range is good
            'alpha': trial.suggest_float('alpha', 0.0, 10.0),  # This range is good
            'random_state': 42,  # Fixed value for reproducibility
            'verbosity': 0  # Silent mode
        }

        # Model definition
        model = XGBClassifier(**params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)


def main(n_trials=100):

    # Load the top 10 results
    top_10_results = pd.read_csv("results/ML/csv/raw_top.csv")

    # Get the top 10 combinations
    combinations = [
        (row["encoding_type"], row["vector_size"], row["model"]) for _, row in top_10_results.iterrows()
    ]

    # Iterate over the top 5 combinations
    for combo in combinations:

        # Unpack the combination
        encoding_type, vector_size, model_name = combo
        study_name = f"{model_name}_{encoding_type}_{vector_size}_2"

        if study_name + ".pkl" not in os.listdir(f"results/ML/finetune"):
            print(f"Fine-tuning {model_name} for {encoding_type} - {vector_size}...")

            # Load train and test data
            train = pd.read_csv(f"data/vectors/{encoding_type}/train_{vector_size}.csv")
            test = pd.read_csv(f"data/vectors/{encoding_type}/test_{vector_size}.csv")

            # Separate the features and the target variable
            y_train = train["label"].astype(float)
            X_train = train.drop(columns=["label"])
            
            y_test = test["label"].astype(float)
            X_test = test.drop(columns=["label"])

            study = optuna.create_study(
                sampler=TPESampler(),
                pruner = HyperbandPruner(
                    min_resource=5,
                    max_resource=500,
                    reduction_factor=5),
                study_name=study_name,
                direction='maximize')

            # Optimize the objective function
            study.optimize(
                lambda trial: objective(trial, X_train, y_train, X_test, y_test, model_name),
                n_trials=n_trials,
                n_jobs=4)

            # Save the study
            joblib.dump(study, f"results/ML/finetune/{study_name}.pkl")

        else:
            print(f"Study {study_name} already exists.")


if __name__ == "__main__":
    main()
