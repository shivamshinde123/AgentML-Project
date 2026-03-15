# ============================================================================
# AGENT FILE - ONLY FILE YOU ARE ALLOWED TO MODIFY
# This is the only file the AI agent is allowed to modify.
# The agent iterates on this file by trying different models, hyperparameters,
# regularization strategies, and ensemble methods.
# ============================================================================

"""
AgentML Training Script

Current experiment: SVC with RBF kernel
Notes: SVC with RBF kernel, C=10, gamma=scale. SVMs often work well on small datasets.
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    f1_score, accuracy_score, mean_squared_error, r2_score,
)


def load_data(path="processed/data_splits.pkl"):
    """Load preprocessed data splits."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_model(task_type):
    """
    Return the model to train.
    The agent modifies this function to try different models and hyperparameters.
    """
    if task_type == "classification":
        model = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            random_state=42,
        )
    else:
        model = SVR(
            kernel="rbf",
            C=10.0,
            gamma="scale",
        )
    return model


def get_scoring(task_type, metric=None):
    """Get the sklearn scoring string for cross-validation."""
    if metric and metric != "auto":
        return metric
    if task_type == "classification":
        return "f1_weighted"
    else:
        return "neg_root_mean_squared_error"


def evaluate_model(model, X_val, y_val, task_type):
    """Evaluate the trained model on the validation set."""
    y_pred = model.predict(X_val)

    if task_type == "classification":
        val_score = f1_score(y_val, y_pred, average="weighted")
    else:
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        val_score = -rmse  # Negative so higher is better (consistent with sklearn)

    return val_score


def train():
    """Run a single training experiment."""
    # Load data
    data = load_data()
    metadata = data["metadata"]
    task_type = metadata["task_type"]
    metric_name = metadata.get("metric", "auto")
    cv_folds = metadata.get("cv_folds", 10)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    # Get model and scoring
    model = get_model(task_type)
    scoring = get_scoring(task_type, metric_name)
    model_name = type(model).__name__

    # Get or create MLflow run
    run_id = os.environ.get("MLFLOW_RUN_ID")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    if run_id:
        run_context = mlflow.start_run(run_id=run_id)
    else:
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "agentml_experiment")
        mlflow.set_experiment(experiment_name)
        run_context = mlflow.start_run()

    with run_context:
        actual_run_id = mlflow.active_run().info.run_id

        # Cross-validation
        start_time = time.time()
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds, scoring=scoring, n_jobs=-1,
        )
        cv_time = time.time() - start_time

        # Train on full training set
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start

        total_time = cv_time + train_time

        # Evaluate on validation set
        val_score = evaluate_model(model, X_val, y_val, task_type)

        # Log to MLflow
        mlflow.log_param("model_name", model_name)
        model_params = model.get_params()
        # MLflow has a param value length limit, so filter long params
        for k, v in model_params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                mlflow.log_param(k, str(v)[:250])

        mlflow.log_metric("cv_mean", float(np.mean(cv_scores)))
        mlflow.log_metric("cv_std", float(np.std(cv_scores)))
        mlflow.log_metric("val_score", float(val_score))
        mlflow.log_metric("training_time", float(total_time))
        mlflow.log_metric("cv_folds", cv_folds)

        # Log the agent's notes about why this change was made
        notes = "SVC with RBF kernel, C=10, gamma=scale"
        mlflow.log_param("agent_notes", notes)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        # Print JSON summary for orchestrator
        result = {
            "run_id": actual_run_id,
            "model_name": model_name,
            "metric_name": metric_name,
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "val_score": float(val_score),
            "training_time": float(total_time),
            "notes": notes,
        }
        print(json.dumps(result))

    return result


if __name__ == "__main__":
    train()
