# ============================================================================
# AGENT FILE - ONLY FILE YOU ARE ALLOWED TO MODIFY
# This is the only file the AI agent is allowed to modify.
# The agent iterates on this file by trying different models, hyperparameters,
# regularization strategies, and ensemble methods.
# ============================================================================

"""
AgentML Training Script

Current experiment: HistGradientBoostingRegressor tuned v1
Notes: Increase max_iter=500, lower lr=0.05, max_depth=8 for better generalization.
"""

import os
import sys
import json
import time
import pickle
import re
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    f1_score, accuracy_score, mean_squared_error, r2_score,
)


def load_data(path="processed/data_splits.pkl"):
    """Load preprocessed data splits."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def parse_amount(val):
    """Parse 'Amount(in rupees)' like '75.9 Lac' or '1.2 Cr' to numeric."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    try:
        num = float(re.findall(r'[\d.]+', s)[0])
    except (IndexError, ValueError):
        return np.nan
    if 'cr' in s:
        return num * 1e7
    elif 'lac' in s or 'lakh' in s:
        return num * 1e5
    return num


def parse_area(val):
    """Parse area strings like '1390 sqft' to numeric."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    try:
        return float(re.findall(r'[\d.]+', s)[0])
    except (IndexError, ValueError):
        return np.nan


def parse_floor(val):
    """Parse '4 out of 4' -> (current_floor, total_floors)."""
    s = str(val).strip().lower()
    match = re.findall(r'(\d+)\s*out\s*of\s*(\d+)', s)
    if match:
        return float(match[0][0]), float(match[0][1])
    nums = re.findall(r'\d+', s)
    if nums:
        return float(nums[0]), np.nan
    return np.nan, np.nan


def parse_car_parking(val):
    """Parse '1 Covered' or '2 Open' -> count."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    nums = re.findall(r'\d+', s)
    if nums:
        return float(nums[0])
    return np.nan


def parse_simple_numeric(val):
    """Try to parse a simple numeric value from string."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    try:
        return float(s)
    except ValueError:
        nums = re.findall(r'[\d.]+', s)
        if nums:
            return float(nums[0])
        return np.nan


# Store fitted label encoders for consistent encoding across train/val
_label_encoders = {}


def engineer_features(X_raw, feature_names, fit=True):
    """Transform raw object array into engineered numeric DataFrame."""
    global _label_encoders
    from sklearn.preprocessing import LabelEncoder

    df = pd.DataFrame(X_raw, columns=feature_names)
    features = {}

    # Numeric features - parse from strings
    if 'Index' in df.columns:
        features['Index'] = pd.to_numeric(df['Index'], errors='coerce')
    if 'Amount(in rupees)' in df.columns:
        features['Amount'] = df['Amount(in rupees)'].apply(parse_amount)
    if 'Carpet Area' in df.columns:
        features['CarpetArea'] = df['Carpet Area'].apply(parse_area)
    if 'Floor' in df.columns:
        floor_parsed = df['Floor'].apply(parse_floor)
        features['CurrentFloor'] = floor_parsed.apply(lambda x: x[0])
        features['TotalFloors'] = floor_parsed.apply(lambda x: x[1])
    if 'Bathroom' in df.columns:
        features['Bathroom'] = df['Bathroom'].apply(parse_simple_numeric)
    if 'Balcony' in df.columns:
        features['Balcony'] = df['Balcony'].apply(parse_simple_numeric)
    if 'Car Parking' in df.columns:
        features['CarParking'] = df['Car Parking'].apply(parse_car_parking)
    if 'Super Area' in df.columns:
        features['SuperArea'] = df['Super Area'].apply(parse_area)
    if 'Dimensions' in df.columns:
        features['Dimensions'] = pd.to_numeric(df['Dimensions'], errors='coerce')
    if 'Plot Area' in df.columns:
        features['PlotArea'] = pd.to_numeric(df['Plot Area'], errors='coerce')
    if 'Title' in df.columns:
        features['BHK'] = df['Title'].astype(str).str.extract(
            r'(\d+)\s*BHK', expand=False
        ).astype(float)

    # Label encode categoricals with consistent encoding
    cat_cols = ['location', 'Status', 'Transaction', 'Furnishing', 'facing',
                'overlooking', 'Society', 'Ownership']
    for col in cat_cols:
        if col not in df.columns:
            continue
        enc_name = f'{col}_enc'
        if fit:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            _label_encoders[col] = le
        else:
            le = _label_encoders.get(col)
            if le is None:
                le = LabelEncoder()
                le.fit(df[col].astype(str))

        known = set(le.classes_)
        col_vals = df[col].astype(str).apply(
            lambda x, k=known, c=le.classes_: x if x in k else c[0]
        )
        features[enc_name] = le.transform(col_vals)

    result = pd.DataFrame(features)
    return result.values.astype(np.float64)


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
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
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
        val_score = -rmse

    return val_score


def train():
    """Run a single training experiment."""
    data = load_data()
    metadata = data["metadata"]
    task_type = metadata["task_type"]
    metric_name = metadata.get("metric", "auto")
    cv_folds = metadata.get("cv_folds", 10)
    feature_names = data.get("feature_names",
                             [f"col_{i}" for i in range(data["X_train"].shape[1])])

    # Engineer features
    X_train = engineer_features(data["X_train"], feature_names, fit=True)
    y_train = np.array(data["y_train"], dtype=np.float64)
    X_val = engineer_features(data["X_val"], feature_names, fit=False)
    y_val = np.array(data["y_val"], dtype=np.float64)

    # Remove rows with NaN target
    train_mask = ~np.isnan(y_train)
    val_mask = ~np.isnan(y_val)
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]

    model = get_model(task_type)
    scoring = get_scoring(task_type, metric_name)
    model_name = type(model).__name__

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

        start_time = time.time()
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds, scoring=scoring, n_jobs=-1,
        )
        cv_time = time.time() - start_time

        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start

        total_time = cv_time + train_time
        val_score = evaluate_model(model, X_val, y_val, task_type)

        mlflow.log_param("model_name", model_name)
        model_params = model.get_params()
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

        notes = "HistGradientBoostingRegressor tuned v1: max_iter=500, lr=0.05, max_depth=8, min_samples_leaf=20. Aiming for better generalization."
        mlflow.log_param("agent_notes", notes)
        mlflow.sklearn.log_model(model, "model")

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
