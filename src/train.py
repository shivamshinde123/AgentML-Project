# ============================================================================
# AGENT FILE - ONLY FILE YOU ARE ALLOWED TO MODIFY
# This is the only file the AI agent is allowed to modify.
# The agent iterates on this file by trying different models, hyperparameters,
# regularization strategies, and ensemble methods.
# ============================================================================

"""
AgentML Training Script

Current experiment: HistGradientBoostingRegressor baseline
Notes: Baseline with clean preprocessed data from prepare.py.
       No preprocessing needed here - data is fully numeric and NaN-free.
"""

import os
import json
import time
import pickle
import numpy as np

# Resolve project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
)


def load_data(path=None):
    """Load preprocessed data splits."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "processed", "data_splits.pkl")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Processed data not found at {path}. "
            "Run prepare.py first to generate the data splits."
        )
    except (pickle.UnpicklingError, EOFError) as e:
        raise RuntimeError(f"Failed to load data splits from {path}: {e}")
    return data


def get_model(task_type):
    """
    Return the model to train.
    The agent modifies this function to try different models and hyperparameters.
    """
    if task_type == "classification":
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
        )
    return model


def get_scoring(task_type, metric=None):
    """Get the sklearn scoring string for cross-validation.

    Uses the metric from program.md if explicitly configured; otherwise
    falls back to f1_weighted for classification and neg_RMSE for regression.
    """
    if metric and metric != "auto":
        return metric
    if task_type == "classification":
        return "f1_weighted"
    else:
        return "neg_root_mean_squared_error"


def evaluate_model(model, X_val, y_val, task_type, metric_name="auto"):
    """
    Evaluate the trained model on the validation set.
    Returns (primary_score, all_metrics_dict).
    Primary score matches the configured metric for fair comparison.
    All metrics are logged to MLflow for full visibility.
    """
    y_pred = model.predict(X_val)

    if task_type == "classification":
        metrics = {
            "val_f1_weighted": f1_score(y_val, y_pred, average="weighted"),
            "val_accuracy": accuracy_score(y_val, y_pred),
            "val_precision_weighted": precision_score(y_val, y_pred, average="weighted", zero_division=0),
            "val_recall_weighted": recall_score(y_val, y_pred, average="weighted", zero_division=0),
        }
        # Map configured metric to the corresponding computed value
        metric_map = {
            "f1_weighted": metrics["val_f1_weighted"],
            "accuracy": metrics["val_accuracy"],
        }
        val_score = metric_map.get(metric_name, metrics["val_f1_weighted"])
    else:
        n = len(y_val)
        p = X_val.shape[1]
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        metrics = {
            "val_rmse": rmse,
            "val_mae": mae,
            "val_r2_adjusted": r2_adj,
        }
        # Map configured metric to the corresponding computed value
        metric_map = {
            "neg_root_mean_squared_error": -rmse,
            "neg_mean_squared_error": -mean_squared_error(y_val, y_pred),
            "r2_adjusted": r2_adj,
        }
        val_score = metric_map.get(metric_name, -rmse)

    return val_score, metrics


def train():
    """Run a single training experiment.

    Loads preprocessed splits, performs k-fold cross-validation on the training
    set, fits the final model, evaluates on the validation set, and logs
    everything (params, metrics, model artifact) to MLflow.

    The result dict is printed as JSON on the last line of stdout so that
    run_experiment.py can parse it from the subprocess output.
    """
    data = load_data()
    metadata = data["metadata"]
    task_type = metadata["task_type"]       # "classification" or "regression"
    metric_name = metadata.get("metric", "auto")
    cv_folds = metadata.get("cv_folds", 10)

    # Data is already fully preprocessed - just load and use directly
    X_train = np.array(data["X_train"], dtype=np.float64)
    y_train = np.array(data["y_train"], dtype=np.float64)
    X_val = np.array(data["X_val"], dtype=np.float64)
    y_val = np.array(data["y_val"], dtype=np.float64)

    model = get_model(task_type)
    scoring = get_scoring(task_type, metric_name)
    model_name = type(model).__name__

    # Attach to the MLflow run created by run_experiment.py (passed via env vars),
    # or start a new standalone run when train.py is executed directly.
    run_id = os.environ.get("MLFLOW_RUN_ID")
    default_mlruns = "file:///" + os.path.normpath(
        os.path.join(PROJECT_ROOT, "mlruns")).replace("\\", "/")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", default_mlruns)
    mlflow.set_tracking_uri(tracking_uri)

    if run_id:
        # Reuse the parent run opened by the orchestrator
        run_context = mlflow.start_run(run_id=run_id)
    else:
        # Standalone execution: open a new run under the configured experiment
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "agentml_experiment")
        mlflow.set_experiment(experiment_name)
        run_context = mlflow.start_run()

    with run_context:
        actual_run_id = mlflow.active_run().info.run_id

        # Cross-validation on training data to estimate generalisation performance
        start_time = time.time()
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds, scoring=scoring, n_jobs=-1,
        )
        cv_time = time.time() - start_time

        # Final fit on the full training set
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start

        total_time = cv_time + train_time

        # Evaluate on the held-out validation set
        val_score, all_metrics = evaluate_model(
            model, X_val, y_val, task_type, metric_name
        )

        # Log model identity and all hyperparameters
        mlflow.log_param("model_name", model_name)
        model_params = model.get_params()
        for k, v in model_params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                # Truncate params that exceed MLflow's 250-char limit
                mlflow.log_param(k, str(v)[:250])

        # Log summary metrics used by the orchestrator for comparison
        mlflow.log_metric("cv_mean", float(np.mean(cv_scores)))
        mlflow.log_metric("cv_std", float(np.std(cv_scores)))
        mlflow.log_metric("val_score", float(val_score))
        mlflow.log_metric("training_time", float(total_time))
        mlflow.log_metric("cv_folds", cv_folds)

        # Log all evaluation metrics for full visibility
        for metric_key, metric_val in all_metrics.items():
            mlflow.log_metric(metric_key, float(metric_val))

        notes = "HGBR baseline: max_iter=500, lr=0.05, depth=6, min_leaf=20, l2=0.1"
        mlflow.log_param("agent_notes", notes)
        # Persist the fitted model as an MLflow artifact for later retrieval
        mlflow.sklearn.log_model(model, "model")

        # Build result dict — printed as JSON so run_experiment.py can parse it
        result = {
            "run_id": actual_run_id,
            "model_name": model_name,
            "metric_name": metric_name,
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "val_score": float(val_score),
            "training_time": float(total_time),
            "notes": notes,
            "all_metrics": {k: float(v) for k, v in all_metrics.items()},
        }
        print(json.dumps(result))

    return result


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n[train] Interrupted by user.")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"[train] ERROR: {e}")
        raise
