# ============================================================================
# READ ONLY - DO NOT MODIFY
# This file is read only for the agent. The agent must never modify this file.
# The agent can change the imputer, encoder, or scaler settings in program.md
# to control which preprocessing strategies are used.
# ============================================================================

"""
AgentML Data Preparation Pipeline

Full-featured preprocessing with configurable strategies for:

Imputation methods:
- median (default): median for numerical, mode for categorical
- mean: mean for numerical, mode for categorical
- knn: KNN imputer for numerical, mode for categorical
- iterative: IterativeImputer (MICE) for numerical, mode for categorical

Encoding methods:
- label_onehot (default): label encoding for binary, one-hot for multi-class
- ordinal: ordinal encoding for all categorical features
- target: target encoding for all categorical features

Normalization/Scaling methods:
- standard (default): StandardScaler
- minmax: MinMaxScaler
- robust: RobustScaler
- maxabs: MaxAbsScaler

CRITICAL: Imputation, encoding, normalization, and class imbalance handling
are ALWAYS executed and cannot be skipped. This ensures train.py receives
clean, fully numeric data and never needs its own preprocessing.
"""

import os
import pickle
import yaml
import numpy as np
import pandas as pd

# Resolve project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    LabelEncoder, OrdinalEncoder,
)
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from imblearn.over_sampling import SMOTE


# Critical steps that can NEVER be skipped
CRITICAL_STEPS = {"duplicates", "imputation", "encoding", "normalization", "smote"}


def parse_program_md(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "program.md")
    """Parse YAML frontmatter from program.md."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"[prepare] Config file not found: {path}. "
            "Ensure program.md exists at the project root."
        )
    except IOError as e:
        raise IOError(f"[prepare] Could not read config file {path}: {e}")
    parts = content.split("---", 2)
    if len(parts) >= 3:
        try:
            config = yaml.safe_load(parts[1])
        except yaml.YAMLError as e:
            raise ValueError(f"[prepare] Failed to parse YAML frontmatter: {e}")
        instructions = parts[2].strip()
    else:
        config = {}
        instructions = content
    return config, instructions


def _parse_numeric_from_string(series):
    """Try to parse numeric values from string columns (e.g., '75.9 Lac', '1390 sqft')."""
    import re

    def _parse_single(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip().lower()
        # Handle Cr/Lac suffixes (Indian number system)
        try:
            num = float(re.findall(r'[\d.]+', s)[0])
        except (IndexError, ValueError):
            return np.nan
        if 'cr' in s:
            return num * 1e7
        elif 'lac' in s or 'lakh' in s:
            return num * 1e5
        return num

    return series.apply(_parse_single)


def _try_convert_to_numeric(X):
    """
    Attempt to convert object columns that contain numeric-like strings to actual numbers.
    This handles cases like '75.9 Lac', '1390 sqft', '4 out of 4', etc.
    """
    import re

    object_cols = X.select_dtypes(include=["object"]).columns.tolist()
    converted_cols = []

    for col in object_cols:
        sample = X[col].dropna().head(50)
        if len(sample) == 0:
            continue

        # Check if most values contain numeric content
        numeric_pattern = re.compile(r'[\d.]+')
        has_numbers = sample.apply(lambda v: bool(numeric_pattern.search(str(v))))
        numeric_ratio = has_numbers.mean()

        if numeric_ratio >= 0.8:
            # This column is mostly numeric strings - parse them
            parsed = _parse_numeric_from_string(X[col])
            non_null_ratio = parsed.notna().mean()
            if non_null_ratio >= 0.5:
                X[col] = parsed
                converted_cols.append(col)

    if converted_cols:
        print(f"[prepare] Converted {len(converted_cols)} string columns to numeric: "
              f"{converted_cols}")

    return X


def load_dataset(config):
    """Load CSV dataset and separate features from target."""
    dataset_config = config.get("dataset", {})
    path = dataset_config.get("path", "data/dataset.csv")
    # Resolve relative dataset paths against project root
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    target_column = dataset_config.get("target_column", "target")

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"[prepare] Dataset not found: {path}. "
            "Check the 'path' setting in program.md."
        )
    except Exception as e:
        raise RuntimeError(f"[prepare] Failed to load dataset from {path}: {e}")
    print(f"[prepare] Loaded dataset: {path} ({df.shape[0]} rows, {df.shape[1]} columns)")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # If target is non-numeric (e.g., "75.9 Lac"), try to parse it
    if y.dtype == "object":
        y_parsed = _parse_numeric_from_string(y)
        successful = y_parsed.notna().sum()
        if successful > 0:
            print(f"[prepare] Parsed target column from strings to numeric "
                  f"({successful}/{len(y)} values)")
            y = y_parsed

    # Drop rows where target is NaN (can't train on missing targets)
    nan_mask = y.isna()
    if nan_mask.any():
        n_dropped = nan_mask.sum()
        X = X[~nan_mask].reset_index(drop=True)
        y = y[~nan_mask].reset_index(drop=True)
        print(f"[prepare] Dropped {n_dropped} rows with missing target values")

    return X, y


def drop_low_value_columns(X):
    """
    Intelligently drop columns that typically don't add ML value (e.g., IDs, names,
    titles, descriptions, URLs) while attempting to extract useful info first.

    Logic:
    1. Detect columns that look like IDs (monotonic integers, or 'id'/'index' in name)
    2. Detect columns that look like free-text (titles, names, descriptions, URLs)
    3. Before dropping, try to extract useful features:
       - From title-like columns: extract numeric patterns (e.g., "3 BHK" -> 3)
       - From name/description columns with few unique values: keep (may be categorical)
    4. Drop the original low-value column after extraction
    """
    import re

    dropped = []
    extracted = {}

    # Patterns that suggest low-value columns
    id_patterns = re.compile(r'(?:^id$|_id$|^index$|^serial|^sr[._\s]?no|^row[_\s]?num)',
                             re.IGNORECASE)
    text_patterns = re.compile(r'(?:^title$|^name$|^description$|^desc$|^url$|^link$|'
                               r'^address$|^comment|^note|^remark|^text$)',
                               re.IGNORECASE)

    for col in X.columns.tolist():
        # --- Check for ID-like columns ---
        if id_patterns.search(col):
            # If it's numeric and mostly unique, it's an ID -> drop
            if X[col].dtype in ['int64', 'int32', 'float64']:
                uniqueness = X[col].nunique() / len(X)
                if uniqueness > 0.9:
                    dropped.append(col)
                    continue
            # String IDs
            elif X[col].dtype == 'object':
                uniqueness = X[col].nunique() / len(X)
                if uniqueness > 0.5:
                    dropped.append(col)
                    continue

        # --- Check for text-like columns ---
        if text_patterns.search(col):
            if X[col].dtype == 'object':
                nunique = X[col].nunique()
                ratio = nunique / len(X)

                # If low cardinality, it's actually categorical -> keep it
                if nunique <= 50:
                    print(f"[prepare] Keeping '{col}' despite text-like name "
                          f"({nunique} unique values, treating as categorical)")
                    continue

                # Try to extract numeric info (e.g., "3 BHK Flat" -> 3)
                sample = X[col].dropna().head(100)
                numeric_matches = sample.astype(str).str.extract(r'(\d+)', expand=False)
                if numeric_matches.notna().mean() > 0.5:
                    extracted_name = f"{col}_extracted_num"
                    X[extracted_name] = X[col].astype(str).str.extract(
                        r'(\d+)', expand=False
                    ).astype(float)
                    extracted[col] = extracted_name
                    print(f"[prepare] Extracted numeric feature '{extracted_name}' "
                          f"from '{col}'")

                dropped.append(col)
                continue

        # --- Check for numeric columns that are actually IDs ---
        # (High uniqueness, monotonic or near-monotonic)
        if X[col].dtype in ['int64', 'int32']:
            uniqueness = X[col].nunique() / len(X)
            if uniqueness > 0.95:
                is_monotonic = X[col].is_monotonic_increasing or X[col].is_monotonic_decreasing
                if is_monotonic:
                    dropped.append(col)
                    continue

        # --- Check for object columns that are essentially unique (free text) ---
        if X[col].dtype == 'object':
            nunique = X[col].nunique()
            ratio = nunique / len(X)
            avg_len = X[col].dropna().astype(str).str.len().mean()

            # Long strings with high uniqueness = free text -> drop
            if ratio > 0.5 and avg_len > 50:
                dropped.append(col)
                continue

    if dropped:
        X = X.drop(columns=dropped)
        print(f"[prepare] Dropped {len(dropped)} low-value columns: {dropped}")
    else:
        print("[prepare] No low-value columns detected")

    return X


def detect_task_type(y, config):
    """Auto-detect whether this is a classification or regression task."""
    task_type = config.get("dataset", {}).get("task_type", "auto")

    if task_type != "auto":
        print(f"[prepare] Task type (from config): {task_type}")
        return task_type

    if y.dtype == "object" or y.dtype.name == "category":
        task_type = "classification"
    elif y.nunique() <= 20 and y.dtype in ["int64", "int32"]:
        task_type = "classification"
    else:
        task_type = "regression"

    print(f"[prepare] Task type (auto-detected): {task_type}")
    return task_type


def remove_duplicates(X, y, skip_steps):
    """Remove duplicate rows. CRITICAL: Always executed."""
    if "duplicates" in skip_steps:
        print("[prepare] WARNING: Duplicate removal is a critical step and cannot be "
              "skipped. Ignoring skip request.")

    combined = X.copy()
    combined["__target__"] = y.values
    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)

    y = combined["__target__"]
    X = combined.drop(columns=["__target__"])

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    print(f"[prepare] Removed {before - after} duplicate rows ({before} -> {after})")
    return X, y


def impute_missing(X, skip_steps, imputer_type="median"):
    """
    Impute missing values using the configured strategy.

    Strategies:
    - median: SimpleImputer with median (numerical), mode (categorical)
    - mean: SimpleImputer with mean (numerical), mode (categorical)
    - knn: KNNImputer for numerical, mode for categorical
    - iterative: IterativeImputer (MICE) for numerical, mode for categorical

    CRITICAL: This step is always executed regardless of skip_steps.
    """
    if "imputation" in skip_steps:
        print("[prepare] WARNING: Imputation is a critical step and cannot be skipped. "
              "Ignoring skip request.")

    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    missing_before = X.isnull().sum().sum()

    # Always impute categorical columns with mode (all strategies)
    for col in categorical_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    # Drop numerical columns that are entirely NaN (no information to impute from)
    all_nan_cols = [col for col in numerical_cols if X[col].isnull().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        numerical_cols = [c for c in numerical_cols if c not in all_nan_cols]
        print(f"[prepare] Dropped {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")

    # Impute numerical columns based on strategy, with fallback chain
    IMPUTER_FALLBACK_ORDER = ["knn", "iterative", "mean", "median"]

    if numerical_cols and X[numerical_cols].isnull().any().any():
        # Build ordered list: requested method first, then fallbacks
        methods_to_try = [imputer_type] + [m for m in IMPUTER_FALLBACK_ORDER
                                            if m != imputer_type]
        succeeded = False

        for method in methods_to_try:
            try:
                if method == "mean":
                    for col in numerical_cols:
                        if X[col].isnull().any():
                            X[col] = X[col].fillna(X[col].mean())
                    label = "mean"

                elif method == "knn":
                    imputer = KNNImputer(n_neighbors=5)
                    X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
                    label = "KNN k=5"

                elif method == "iterative":
                    imputer = IterativeImputer(max_iter=10, random_state=42)
                    X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
                    label = "IterativeImputer/MICE"

                else:  # median
                    for col in numerical_cols:
                        if X[col].isnull().any():
                            X[col] = X[col].fillna(X[col].median())
                    label = "median"

                if method != imputer_type:
                    print(f"[prepare] WARNING: '{imputer_type}' imputation failed, "
                          f"fell back to '{method}'")
                print(f"[prepare] Imputed {missing_before} missing values "
                      f"(numerical: {label}, categorical: mode)")
                succeeded = True
                break

            except Exception as e:
                print(f"[prepare] Imputation method '{method}' failed: {e}")
                continue

        if not succeeded:
            raise RuntimeError("[prepare] CRITICAL: All imputation methods failed.")
    else:
        print(f"[prepare] Imputed {missing_before} missing values "
              f"({len(numerical_cols)} numerical cols, "
              f"{len(categorical_cols)} categorical cols with mode)")

    return X


def encode_categoricals(X, y, task_type, skip_steps, encoder_type="label_onehot"):
    """
    Encode categorical features using the configured strategy.

    Strategies:
    - label_onehot (default): label encoding for binary, one-hot for multi-class
    - ordinal: OrdinalEncoder for all categorical features
    - target: target/mean encoding for all categorical features

    CRITICAL: This step is always executed regardless of skip_steps.
    Also handles target variable encoding for classification tasks.
    """
    if "encoding" in skip_steps:
        print("[prepare] WARNING: Encoding is a critical step and cannot be skipped. "
              "Ignoring skip request.")

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}

    if len(categorical_cols) == 0:
        print("[prepare] No categorical features to encode")
    else:
        ENCODER_FALLBACK_ORDER = ["label_onehot", "ordinal", "target"]
        methods_to_try = [encoder_type] + [m for m in ENCODER_FALLBACK_ORDER
                                            if m != encoder_type]
        succeeded = False

        for method in methods_to_try:
            try:
                # Work on a copy so failed attempts don't corrupt X
                X_attempt = X.copy()
                encoders_attempt = {}

                if method == "ordinal":
                    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                    X_attempt[categorical_cols] = oe.fit_transform(
                        X_attempt[categorical_cols].astype(str))
                    encoders_attempt["__ordinal__"] = {
                        "type": "ordinal", "encoder": oe, "columns": categorical_cols}
                    label = f"Ordinal encoded {len(categorical_cols)} categorical columns"

                elif method == "target":
                    for col in categorical_cols:
                        if task_type == "classification":
                            y_numeric = (y.astype(float) if y.dtype != "object"
                                         else pd.factorize(y)[0])
                            mapping = pd.Series(y_numeric).groupby(
                                X_attempt[col].astype(str)).mean()
                        else:
                            mapping = y.groupby(X_attempt[col].astype(str)).mean()

                        global_mean = mapping.mean()
                        X_attempt[col] = X_attempt[col].astype(str).map(mapping).fillna(
                            global_mean)
                        encoders_attempt[col] = {
                            "type": "target", "mapping": mapping.to_dict(),
                            "global_mean": global_mean}
                    label = f"Target encoded {len(categorical_cols)} categorical columns"

                else:  # label_onehot (default)
                    MAX_ONEHOT_CARDINALITY = 20

                    binary_cols = [c for c in categorical_cols
                                   if X_attempt[c].nunique() <= 2]
                    low_card_cols = [c for c in categorical_cols
                                     if 2 < X_attempt[c].nunique() <= MAX_ONEHOT_CARDINALITY]
                    high_card_cols = [c for c in categorical_cols
                                      if X_attempt[c].nunique() > MAX_ONEHOT_CARDINALITY]

                    for col in binary_cols:
                        le = LabelEncoder()
                        X_attempt[col] = le.fit_transform(X_attempt[col].astype(str))
                        encoders_attempt[col] = {"type": "label", "encoder": le}

                    if low_card_cols:
                        X_attempt = pd.get_dummies(
                            X_attempt, columns=low_card_cols, drop_first=False, dtype=int)
                        encoders_attempt["__onehot_cols__"] = low_card_cols

                    for col in high_card_cols:
                        le = LabelEncoder()
                        X_attempt[col] = le.fit_transform(X_attempt[col].astype(str))
                        encoders_attempt[col] = {"type": "label", "encoder": le}

                    label = (f"Encoded {len(binary_cols)} binary (label), "
                             f"{len(low_card_cols)} low-card (one-hot), "
                             f"{len(high_card_cols)} high-card (label)")

                # Success — commit the attempt
                X = X_attempt
                encoders = encoders_attempt
                if method != encoder_type:
                    print(f"[prepare] WARNING: '{encoder_type}' encoding failed, "
                          f"fell back to '{method}'")
                print(f"[prepare] {label}")
                succeeded = True
                break

            except Exception as e:
                print(f"[prepare] Encoding method '{method}' failed: {e}")
                continue

        if not succeeded:
            raise RuntimeError("[prepare] CRITICAL: All encoding methods failed.")

    # Always encode target if categorical (classification)
    target_encoder = None
    if task_type == "classification" and y.dtype == "object":
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y), name=y.name)
        encoders["__target__"] = {"type": "label", "encoder": target_encoder}
        print(f"[prepare] Encoded target variable: {list(target_encoder.classes_)}")

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
        print(f"[prepare] Unknown scaler '{scaler_type}', falling back to StandardScaler")
        scaler_type = "standard"

    return scalers[scaler_type](), scaler_type


def normalize(X, skip_steps, scaler_type="standard"):
    """
    Apply normalization using the configured scaler.

    CRITICAL: This step is always executed regardless of skip_steps.
    """
    if "normalization" in skip_steps:
        print("[prepare] WARNING: Normalization is a critical step and cannot be skipped. "
              "Ignoring skip request.")

    numerical_cols = X.select_dtypes(include=["number"]).columns

    if len(numerical_cols) == 0:
        print("[prepare] No numerical features to normalize")
        return X, None, scaler_type

    SCALER_FALLBACK_ORDER = ["standard", "minmax", "robust", "maxabs"]
    methods_to_try = [scaler_type] + [m for m in SCALER_FALLBACK_ORDER
                                       if m != scaler_type]

    for method in methods_to_try:
        try:
            scaler, actual_type = get_scaler(method)
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
            if method != scaler_type:
                print(f"[prepare] WARNING: '{scaler_type}' scaler failed, "
                      f"fell back to '{method}'")
            print(f"[prepare] Applied {actual_type} scaler ({type(scaler).__name__}) "
                  f"to {len(numerical_cols)} numerical features")
            return X, scaler, actual_type

        except Exception as e:
            print(f"[prepare] Scaler '{method}' failed: {e}")
            continue

    raise RuntimeError("[prepare] CRITICAL: All normalization methods failed.")


def handle_imbalance(X, y, task_type, skip_steps, threshold=0.2):
    """
    Apply SMOTE if class imbalance ratio exceeds threshold (classification only).

    CRITICAL: This step is always evaluated regardless of skip_steps.
    For regression tasks, this is a no-op.
    """
    if "smote" in skip_steps:
        print("[prepare] WARNING: Class imbalance handling is a critical step and "
              "cannot be skipped. Ignoring skip request.")

    if task_type != "classification":
        print("[prepare] Skipping SMOTE (not a classification task)")
        return X, y

    class_counts = y.value_counts()
    minority_count = class_counts.min()
    majority_count = class_counts.max()
    imbalance_ratio = minority_count / majority_count

    if imbalance_ratio >= (1 - threshold):
        print(f"[prepare] No class imbalance detected (ratio={imbalance_ratio:.3f}), "
              f"skipping SMOTE")
        return X, y

    print(f"[prepare] Class imbalance detected (ratio={imbalance_ratio:.3f}), "
          f"applying SMOTE")

    min_samples = class_counts.min()
    k_neighbors = min(5, min_samples - 1)
    if k_neighbors < 1:
        print("[prepare] Too few minority samples for SMOTE, skipping")
        return X, y

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"[prepare] SMOTE resampling: {len(X)} -> {len(X_resampled)} samples")
    X = pd.DataFrame(X_resampled, columns=X.columns)
    y = pd.Series(y_resampled, name=y.name)
    return X, y


def select_features(X, y, task_type, skip_steps, max_features=20):
    """Apply SelectKBest if number of features exceeds max_features."""
    if "feature_selection" in skip_steps:
        print("[prepare] Skipping feature selection (as configured)")
        return X, None

    if X.shape[1] <= max_features:
        print(f"[prepare] {X.shape[1]} features <= {max_features} threshold, "
              f"skipping feature selection")
        return X, None

    print(f"[prepare] {X.shape[1]} features > {max_features} threshold, "
          f"applying SelectKBest (k={max_features})")

    score_func = f_classif if task_type == "classification" else f_regression
    selector = SelectKBest(score_func=score_func, k=max_features)
    X_selected = selector.fit_transform(X, y)

    selected_mask = selector.get_support()
    selected_columns = X.columns[selected_mask].tolist()
    X = pd.DataFrame(X_selected, columns=selected_columns)

    print(f"[prepare] Selected {len(selected_columns)} features: {selected_columns}")
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

    print(f"[prepare] Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def verify_no_remaining_objects(X, stage="final"):
    """
    Safety check: ensure no object/category columns remain after preprocessing.
    This guarantees train.py receives fully numeric data.
    """
    object_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if object_cols:
        raise RuntimeError(
            f"[prepare] CRITICAL: {len(object_cols)} non-numeric columns remain "
            f"after {stage} preprocessing: {object_cols}. "
            f"These must be encoded before passing to train.py."
        )


def save_processed(X_train, X_val, X_test, y_train, y_val, y_test,
                   feature_names, metadata, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "processed")
    """Save processed data splits to pickle file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"[prepare] Could not create output directory {output_dir}: {e}")

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

    try:
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
    except IOError as e:
        raise RuntimeError(f"[prepare] Could not save processed data to {output_path}: {e}")

    print(f"[prepare] Saved processed data to {output_path}")
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


def sanitize_skip_steps(skip_steps):
    """Remove critical steps from skip_steps and warn the user."""
    blocked = [s for s in skip_steps if s in CRITICAL_STEPS]
    if blocked:
        print(f"[prepare] WARNING: The following critical steps cannot be skipped "
              f"and will be executed anyway: {blocked}")
    # Only allow non-critical steps to be skipped (duplicates, feature_selection)
    return [s for s in skip_steps if s not in CRITICAL_STEPS]


def main(config_path=None):
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "program.md")
    """Run the full preprocessing pipeline with alternative strategy support."""
    print("=" * 60)
    print("AgentML Data Preparation Pipeline")
    print("=" * 60)

    # Parse config
    config, instructions = parse_program_md(config_path)
    raw_skip_steps = config.get("preprocessing", {}).get("skip_steps", [])
    skip_steps = sanitize_skip_steps(raw_skip_steps)
    scaler_type = config.get("preprocessing", {}).get("scaler", "standard")
    imputer_type = config.get("preprocessing", {}).get("imputer", "median")
    encoder_type = config.get("preprocessing", {}).get("encoder", "label_onehot")

    print(f"[prepare] Config: imputer={imputer_type}, encoder={encoder_type}, "
          f"scaler={scaler_type}")

    # Load dataset
    X, y = load_dataset(config)

    # Detect task type
    task_type = detect_task_type(y, config)

    # Pipeline steps (critical steps are ALWAYS executed)
    X, y = remove_duplicates(X, y, skip_steps)

    # Drop low-value columns (IDs, free text, etc.) with intelligent extraction
    X = drop_low_value_columns(X)

    # Convert numeric-like string columns BEFORE imputation so NaNs from
    # failed conversions get imputed alongside original missing values
    X = _try_convert_to_numeric(X)

    X = impute_missing(X, skip_steps, imputer_type)
    X, y, encoders = encode_categoricals(X, y, task_type, skip_steps, encoder_type)

    # Second imputation pass: encoding (e.g., target encoding) or numeric
    # conversion may have introduced new NaNs. Fill any remaining NaNs.
    remaining_nans = X.isnull().sum().sum()
    if remaining_nans > 0:
        print(f"[prepare] Post-encoding cleanup: filling {remaining_nans} remaining NaNs")
        numerical_cols = X.select_dtypes(include=["number"]).columns
        for col in numerical_cols:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)

    X, scaler, actual_scaler_type = normalize(X, skip_steps, scaler_type)

    # Verify all data is numeric before SMOTE and downstream steps
    verify_no_remaining_objects(X, stage="post-encoding")

    X, y = handle_imbalance(X, y, task_type, skip_steps)
    X, selector = select_features(X, y, task_type, skip_steps)

    # Final verification: ensure fully numeric output
    verify_no_remaining_objects(X, stage="final")

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, task_type)

    # Build metadata
    metric_name = get_metric_name(task_type, config)
    metadata = {
        "task_type": task_type,
        "metric": metric_name,
        "target_column": config.get("dataset", {}).get("target_column", "target"),
        "imputer_type": imputer_type,
        "encoder_type": encoder_type,
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
    print("[prepare] Preprocessing complete!")
    print(f"  Task type:    {task_type}")
    print(f"  Metric:       {metric_name}")
    print(f"  Imputer:      {imputer_type}")
    print(f"  Encoder:      {encoder_type}")
    print(f"  Scaler:       {actual_scaler_type}")
    print(f"  Features:     {len(feature_names)}")
    print(f"  Train size:   {len(X_train)}")
    print(f"  Val size:     {len(X_val)}")
    print(f"  Test size:    {len(X_test)}")
    print("=" * 60)

    return metadata


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[prepare] Interrupted by user.")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"[prepare] ERROR: {e}")
        raise
