import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, classification_report)
from xgboost import XGBClassifier

DATASET_PATH = "/workspaces/AutoML-Pipeline-for-Tabular-Data-with-MLflow/datasets/titanic_cleaned.csv" 
TARGET_COLUMN = "survived"          
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "AutoML_Pipeline_on_Titanic_with_MLflow"
BEST_MODEL_PATH = "best_model.pkl"


def load_data(path, target):
    """
    Reads the cleaned csv file and splits into features (X) and target label (y).

    Args:
        path (str): Path to the cleaned dataset csv file
        target (str): Name of the target column in the dataset

    Returns:
        X (DataFrame): Features dataframe
        y (Series): Target label series
    """
    try:
        df = pd.read_csv(path)
        print(f"   shape: {df.shape}")
        print(f"   columns: {list(df.columns)}")

        X = df.drop(columns=[target])
        y = df[target]
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def get_models_and_grids():
    """
    Defines the candidate models and their hyperparameter search spaces.

    Returns:
        models (dict): Dictionary with model names as keys and dicts containing:
            - "estimator": The model instance
            - "param_grid": The hyperparameter search space
    """
    models = {
        "LogisticRegression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__solver": ["lbfgs", "liblinear"],
            },
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42),
            "param_grid": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                eval_metric="logloss",
                random_state=42
            ),
            "param_grid": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1, 0.2],
            },
        },
    }
    return models

def train_and_log(model_name, estimator, param_grid, X_train, X_test, y_train, y_test):
    """
    Trains a model, performs hyperparameter tuning using GridSearchCV, and logs parameters, metrics, and artifacts to MLflow.
    
    Args:
        model_name (str): Name of the model, used as the MLflow run name
        estimator (sklearn estimator): An unfitted sklearn-compatible classifier
        param_grid (dict): Hyperparameter grid to search over
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training labels
        y_test (pd.Series): Test labels

    Returns:
        dict: Contains model_name, pipeline, best_params, accuracy, f1_score, roc_auc, and run_id
    """

    print(f"Training: {model_name}")

    # pipeline
    pipeline = Pipeline(steps=[("model", estimator)])

    # GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )

    with mlflow.start_run(run_name=model_name):

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # predictions
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:,1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        roc = roc_auc_score(y_test, y_proba)

        print(f"   Best Params: {grid_search.best_params_}")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   ROC-AUC: {roc:.4f}")

        # log to MLflow
        mlflow.log_param("model_name", model_name)
        clean_params={
            k.replace("model__", ""): v
            for k, v in grid_search.best_params_.items()
        }
        mlflow.log_params(clean_params)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("cv_best_score", grid_search.best_score_)

        # Log classification report as artifact
        report_path = f"{model_name}_report.txt"
        with open(report_path, "w") as fp:
            fp.write(f"Model      : {model_name}\n")
            fp.write(f"Best Params: {grid_search.best_params_}\n\n")
            fp.write(f"Accuracy   : {acc:.4f}\n")
            fp.write(f"F1 Score   : {f1:.4f}\n")
            fp.write(f"ROC-AUC    : {roc:.4f}\n\n")
            fp.write("Classification Report:\n")
            fp.write(classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"]))
        mlflow.log_artifact(report_path)
        # cleanup local report file
        os.remove(report_path)

        # Log model artifact
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id

    return {
        "model_name": model_name,
        "pipeline": best_model,
        "best_params": grid_search.best_params_,
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": roc,
        "run_id": run_id,
    }

def save_best_model(results):
    """
    Saves the best model based on F1 score to a pickle file.

    Args:
        results (list): List of dicts returned by train_and_log for each model

    Returns:
        best (dict): The best model's result dictionary
    """
    best = max(results, key=lambda x: x["f1_score"])
    print(f"\nBest Model: {best['model_name']} (F1={best['f1_score']:.4f})")

    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(best["pipeline"], f)
    print(f"💾 Best model saved to: {BEST_MODEL_PATH}")

    return best

def main():
    """
    Main function to execute the training pipeline:
        1. Set up MLflow tracking
        2. Load data
        3. Train/test split
        4. Define models and hyperparameter grids
        5. Train each model, perform hyperparameter tuning, and log to MLflow
        6. Compare results and save the best model as a pickle file
    """
    # MLflow setup
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment name: {EXPERIMENT_NAME}")

    # load data
    X, y = load_data(DATASET_PATH, TARGET_COLUMN)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

    # Get models
    models = get_models_and_grids()

    # train and log all models
    results = []
    for model_name, config in models.items():
        result = train_and_log(
            model_name=model_name,
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(result)

    # Save best model
    best = save_best_model(results)

    print(f"Open MLflow UI : {MLFLOW_TRACKING_URI}")
    print(f"Best model pkl : {BEST_MODEL_PATH}")
    print(f"Best model run_id: {best['run_id']}")


if __name__ == "__main__":
    main()