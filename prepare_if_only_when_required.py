# ============================================================================
# READ ONLY - DO NOT MODIFY
# This file is read only for the agent. The agent must never modify this file.
# The agent can only choose to switch to this file by changing the scaler
# setting in program.md. Use this only if StandardScaler from prepare.py is
# causing consistently poor performance (no improvement over 3 consecutive
# experiments). First choice should always be prepare.py.
# ============================================================================

"""
AgentML Alternative Data Preparation Pipeline

Contains everything that prepare.py has PLUS alternative scaling strategies:
- MinMaxScaler
- RobustScaler
- MaxAbsScaler

The agent should only switch to this file if the default StandardScaler from
prepare.py is causing consistently poor performance across multiple experiments.
"""

import os
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    LabelEncoder, OneHotEncoder,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from imblearn.over_sampling import SMOTE


def parse_program_md(path="program.md"):
    """Parse YAML frontmatter from program.md."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = content.split("---", 2)
    if len(parts) >= 3:
        config = yaml.safe_load(parts[1])
        instructions = parts[2].strip()
    else:
        config = {}
        instructions = content
    return config, instructions


def load_dataset(config):
    """Load CSV dataset and separate features from target."""
    dataset_config = config.get("dataset", {})
    path = dataset_config.get("path", "data/dataset.csv")
    target_column = dataset_config.get("target_column", "target")

    df = pd.read_csv(path)
    print(f"[prepare_alt] Loaded dataset: {path} ({df.shape[0]} rows, {df.shape[1]} columns)")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def detect_task_type(y, config):
    """Auto-detect whether this is a classification or regression task."""
    task_type = config.get("dataset", {}).get("task_type", "auto")

    if task_type != "auto":
        print(f"[prepare_alt] Task type (from config): {task_type}")
        return task_type

    if y.dtype == "object" or y.dtype.name == "category":
        task_type = "classification"
    elif y.nunique() <= 20 and y.dtype in ["int64", "int32"]:
        task_type = "classification"
    else:
        task_type = "regression"

    print(f"[prepare_alt] Task type (auto-detected): {task_type}")
    return task_type


def remove_duplicates(X, y, skip_steps):
    """Remove duplicate rows."""
    if "duplicates" in skip_steps:
        print("[prepare_alt] Skipping duplicate removal (as configured)")
        return X, y

    combined = X.copy()
    combined["__target__"] = y.values
    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)

    y = combined["__target__"]
    X = combined.drop(columns=["__target__"])

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    print(f"[prepare_alt] Removed {before - after} duplicate rows ({before} -> {after})")
    return X, y


def impute_missing(X, skip_steps):
    """Impute missing values: median for numerical, mode for categorical."""
    if "imputation" in skip_steps:
        print("[prepare_alt] Skipping missing value imputation (as configured)")
        return X

    numerical_cols = X.select_dtypes(include=["number"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    missing_before = X.isnull().sum().sum()

    for col in numerical_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    for col in categorical_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    print(f"[prepare_alt] Imputed {missing_before} missing values "
          f"({len(numerical_cols)} numerical cols with median, "
          f"{len(categorical_cols)} categorical cols with mode)")
    return X


def encode_categoricals(X, y, task_type, skip_steps):
    """
    Encode categorical features:
    - Label encoding for binary categorical columns
    - One-hot encoding for multi-class categorical columns
    Also encode target variable if it's categorical (classification).
    """
    if "encoding" in skip_steps:
        print("[prepare_alt] Skipping categorical encoding (as configured)")
        return X, y, {}

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}

    if len(categorical_cols) == 0:
        print("[prepare_alt] No categorical features to encode")
    else:
        binary_cols = [col for col in categorical_cols if X[col].nunique() <= 2]
        multi_cols = [col for col in categorical_cols if X[col].nunique() > 2]

        for col in binary_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = {"type": "label", "encoder": le}

        if multi_cols:
            X = pd.get_dummies(X, columns=multi_cols, drop_first=False, dtype=int)
            encoders["__onehot_cols__"] = multi_cols

        print(f"[prepare_alt] Encoded {len(binary_cols)} binary cols (label encoding), "
              f"{len(multi_cols)} multi-class cols (one-hot encoding)")

    target_encoder = None
    if task_type == "classification" and y.dtype == "object":
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y), name=y.name)
        encoders["__target__"] = {"type": "label", "encoder": target_encoder}
        print(f"[prepare_alt] Encoded target variable: {list(target_encoder.classes_)}")

    return X, y, encoders


def get_scaler(scaler_type):
    """Get the appropriate scaler based on configuration."""
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "maxabs": MaxAbsScaler,
    }

    if scaler_type not in scalers:
        print(f"[prepare_alt] Unknown scaler '{scaler_type}', falling back to StandardScaler")
        scaler_type = "standard"

    return scalers[scaler_type](), scaler_type


def normalize(X, skip_steps, scaler_type="standard"):
    """Apply normalization using the configured scaler."""
    if "normalization" in skip_steps:
        print("[prepare_alt] Skipping normalization (as configured)")
        return X, None, scaler_type

    scaler, actual_type = get_scaler(scaler_type)
    numerical_cols = X.select_dtypes(include=["number"]).columns

    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    print(f"[prepare_alt] Applied {actual_type} scaler ({type(scaler).__name__}) "
          f"to {len(numerical_cols)} numerical features")
    return X, scaler, actual_type


def handle_imbalance(X, y, task_type, skip_steps, threshold=0.2):
    """Apply SMOTE if class imbalance ratio exceeds threshold (classification only)."""
    if "smote" in skip_steps:
        print("[prepare_alt] Skipping SMOTE (as configured)")
        return X, y

    if task_type != "classification":
        print("[prepare_alt] Skipping SMOTE (not a classification task)")
        return X, y

    class_counts = y.value_counts()
    minority_count = class_counts.min()
    majority_count = class_counts.max()
    imbalance_ratio = minority_count / majority_count

    if imbalance_ratio >= (1 - threshold):
        print(f"[prepare_alt] No class imbalance detected (ratio={imbalance_ratio:.3f}), "
              f"skipping SMOTE")
        return X, y

    print(f"[prepare_alt] Class imbalance detected (ratio={imbalance_ratio:.3f}), "
          f"applying SMOTE")

    min_samples = class_counts.min()
    k_neighbors = min(5, min_samples - 1)
    if k_neighbors < 1:
        print("[prepare_alt] Too few minority samples for SMOTE, skipping")
        return X, y

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"[prepare_alt] SMOTE resampling: {len(X)} -> {len(X_resampled)} samples")
    X = pd.DataFrame(X_resampled, columns=X.columns)
    y = pd.Series(y_resampled, name=y.name)
    return X, y


def select_features(X, y, task_type, skip_steps, max_features=20):
    """Apply SelectKBest if number of features exceeds max_features."""
    if "feature_selection" in skip_steps:
        print("[prepare_alt] Skipping feature selection (as configured)")
        return X, None

    if X.shape[1] <= max_features:
        print(f"[prepare_alt] {X.shape[1]} features <= {max_features} threshold, "
              f"skipping feature selection")
        return X, None

    print(f"[prepare_alt] {X.shape[1]} features > {max_features} threshold, "
          f"applying SelectKBest (k={max_features})")

    score_func = f_classif if task_type == "classification" else f_regression
    selector = SelectKBest(score_func=score_func, k=max_features)
    X_selected = selector.fit_transform(X, y)

    selected_mask = selector.get_support()
    selected_columns = X.columns[selected_mask].tolist()
    X = pd.DataFrame(X_selected, columns=selected_columns)

    print(f"[prepare_alt] Selected {len(selected_columns)} features: {selected_columns}")
    return X, selector


def split_data(X, y, task_type):
    """Split data into 80/10/10 train/validation/test sets."""
    stratify_y = y if task_type == "classification" else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_y
    )

    stratify_temp = y_temp if task_type == "classification" else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=stratify_temp
    )

    print(f"[prepare_alt] Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed(X_train, X_val, X_test, y_train, y_val, y_test,
                   feature_names, metadata, output_dir="processed"):
    """Save processed data splits to pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "data_splits.pkl")

    data = {
        "X_train": X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
        "y_train": y_train.values if isinstance(y_train, pd.Series) else y_train,
        "X_val": X_val.values if isinstance(X_val, pd.DataFrame) else X_val,
        "y_val": y_val.values if isinstance(y_val, pd.Series) else y_val,
        "X_test": X_test.values if isinstance(X_test, pd.DataFrame) else X_test,
        "y_test": y_test.values if isinstance(y_test, pd.Series) else y_test,
        "feature_names": feature_names,
        "metadata": metadata,
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"[prepare_alt] Saved processed data to {output_path}")
    return output_path


def get_metric_name(task_type, config):
    """Determine the primary metric based on task type and config."""
    metric = config.get("metric", {}).get("primary", "auto")
    if metric == "auto":
        if task_type == "classification":
            return "f1_weighted"
        else:
            return "neg_root_mean_squared_error"
    return metric


def main(config_path="program.md"):
    """Run the full preprocessing pipeline with alternative scaler support."""
    print("=" * 60)
    print("AgentML Alternative Data Preparation Pipeline")
    print("=" * 60)

    # Parse config
    config, instructions = parse_program_md(config_path)
    skip_steps = config.get("preprocessing", {}).get("skip_steps", [])
    scaler_type = config.get("preprocessing", {}).get("scaler", "standard")

    # Load dataset
    X, y = load_dataset(config)

    # Detect task type
    task_type = detect_task_type(y, config)

    # Pipeline steps
    X, y = remove_duplicates(X, y, skip_steps)
    X = impute_missing(X, skip_steps)
    X, y, encoders = encode_categoricals(X, y, task_type, skip_steps)
    X, scaler, actual_scaler_type = normalize(X, skip_steps, scaler_type)
    X, y = handle_imbalance(X, y, task_type, skip_steps)
    X, selector = select_features(X, y, task_type, skip_steps)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, task_type)

    # Build metadata
    metric_name = get_metric_name(task_type, config)
    metadata = {
        "task_type": task_type,
        "metric": metric_name,
        "target_column": config.get("dataset", {}).get("target_column", "target"),
        "scaler_type": actual_scaler_type,
        "num_features": X_train.shape[1],
        "num_train_samples": len(X_train),
        "num_val_samples": len(X_val),
        "num_test_samples": len(X_test),
        "num_classes": int(y.nunique()) if task_type == "classification" else None,
        "encoders": encoders,
        "scaler": scaler,
        "selector": selector,
        "cv_folds": config.get("constraints", {}).get("cv_folds", 10),
        "mlflow_config": config.get("mlflow", {}),
    }

    feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else \
        [f"feature_{i}" for i in range(X_train.shape[1])]

    # Save
    save_processed(X_train, X_val, X_test, y_train, y_val, y_test,
                   feature_names, metadata)

    print("=" * 60)
    print("[prepare_alt] Preprocessing complete!")
    print(f"  Task type:    {task_type}")
    print(f"  Metric:       {metric_name}")
    print(f"  Scaler:       {actual_scaler_type}")
    print(f"  Features:     {len(feature_names)}")
    print(f"  Train size:   {len(X_train)}")
    print(f"  Val size:     {len(X_val)}")
    print(f"  Test size:    {len(X_test)}")
    print("=" * 60)

    return metadata


if __name__ == "__main__":
    main()
