# AgentML - Codebase Understanding Guide

This document provides a detailed explanation of how the AgentML codebase works, how the files interact, and the complete flow of an experiment from start to finish.

---

## High-Level Architecture

AgentML is an autonomous ML experimentation loop. The core idea is simple: an AI agent repeatedly edits a single training file (`src/train.py`), runs experiments, and lets the orchestrator decide whether to keep or discard each change based on whether the validation metric improved.

All source code lives in the `src/` directory. Configuration files (`program.md`, `best_scores.json`) and data directories (`data/`, `processed/`) remain at the project root. Each source file uses a `PROJECT_ROOT` constant to resolve paths relative to the project root.

```
                    +------------------+
                    |   program.md     |  <-- User fills in config (project root)
                    +--------+---------+
                             |
                             v
                    +---------------------+
                    | src/run_experiment.py|  <-- Orchestrator (entry point)
                    +--------+------------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
     +------------------+         +------------------+
     | src/prepare.py   |         |  src/train.py    |
     | (preprocessing)  |         |  (agent edits)   |
     +-------+----------+         +--------+---------+
             |                             |
             v                             v
     +----------------+           +------------------+
     |  processed/    |           |     MLflow       |
     | data_splits.pkl|           |  (mlruns/)       |
     +----------------+           +------------------+
                                           |
                                           v
                                  +------------------+
                                  | best_scores.json |
                                  +------------------+
                                           |
                          +----------------+----------------+
                          |                                 |
                          v                                 v
                 +--------------------+     +-----------------------------+
                 | src/select_model.py|     | src/visualization/          |
                 | (promote models)   |     |   analysis.ipynb            |
                 +--------------------+     +-----------------------------+
```

---

## File-by-File Breakdown

### program.md - The Configuration File

**Purpose:** Single source of truth for all experiment settings. The user fills this in before running anything.

**Structure:** Uses YAML frontmatter (between `---` delimiters) for machine-readable config, followed by markdown sections for free-form instructions.

**YAML frontmatter fields:**
- `dataset.path` - Path to the CSV file (e.g., `data/iris.csv`)
- `dataset.target_column` - Name of the column to predict
- `dataset.task_type` - `classification`, `regression`, or `auto` (auto-detects based on target column characteristics)
- `metric.primary` - `auto` (F1 weighted for classification, RMSE for regression) or a specific sklearn scoring string
- `preprocessing.skip_steps` - List of preprocessing steps to skip. Only `feature_selection` can be skipped. Critical steps (`duplicates`, `imputation`, `encoding`, `normalization`, `smote`) are always enforced and cannot be skipped.
- `preprocessing.imputer` - `median` (default), `mean`, `knn`, or `iterative`. Controls the imputation strategy for missing numerical values.
- `preprocessing.encoder` - `label_onehot` (default), `ordinal`, or `target`. Controls how categorical features are encoded.
- `preprocessing.scaler` - `standard` (default), `minmax`, `robust`, or `maxabs`
- `mlflow.experiment_name` - Name for the MLflow experiment
- `mlflow.tracking_uri` - Where MLflow stores data (default: `./mlruns`)
- `mlflow.top_n_models` - How many top models to keep in the registry
- `constraints.cv_folds` - Number of cross-validation folds (default: 10)
- `constraints.max_training_time` - Max training time in seconds

**Markdown sections:**
- Model Constraints - Free-form instructions like "go wild" or "only tree based models"
- Preprocessing Overrides - Notes on what to skip
- Experiment Instructions - Additional guidance like "focus on reducing overfitting"
- Analysis - Which experiment to analyze in the notebook

**Who reads it:** `src/prepare.py`, `src/run_experiment.py`, `src/select_model.py`, and `src/visualization/analysis.ipynb` all parse this file.

**How it's parsed:** Every file that reads it uses the same `parse_program_md()` function pattern - splits on `---`, parses the middle section as YAML via `yaml.safe_load()`, and returns `(config_dict, instructions_string)`.

---

### src/prepare.py - Data Preparation Pipeline

**Purpose:** Takes a raw CSV dataset and produces clean, split, ready-to-train data. Full-featured preprocessing with configurable strategies for imputation, encoding, and normalization. Ensures all critical steps always run so that `train.py` receives fully clean, numeric data with zero NaNs.

**Access:** READ ONLY. The agent can change the `imputer`, `encoder`, or `scaler` fields in `program.md` to control strategies, but cannot modify this file.

**When to change strategies:** When the agent has run 3+ consecutive experiments with no improvement, it should consider changing preprocessing strategies in `program.md`.

**Input:** Raw CSV from `data/` folder, configuration from `program.md`.

**Output:** `processed/data_splits.pkl` - A pickle file containing:
- `X_train`, `y_train` - Training features and labels (numpy arrays)
- `X_val`, `y_val` - Validation features and labels
- `X_test`, `y_test` - Test features and labels
- `feature_names` - List of feature column names
- `metadata` - Dictionary with task type, metric name, imputer/encoder/scaler type, number of classes, cv folds, encoders, and other settings

**Pipeline steps (in order):**

1. **Load dataset** (`load_dataset`) - Reads the CSV, separates features (X) from target (y). Parses non-numeric target columns (e.g., "75.9 Lac" → numeric). Drops rows with missing targets.

2. **Detect task type** (`detect_task_type`) - If `task_type` is `auto`: checks if target is object/category type (classification), integer with <= 20 unique values (classification), or continuous (regression).

3. **Remove duplicates** (`remove_duplicates`) - Combines X and y, drops duplicate rows, resets indices. Critical step — cannot be skipped.

4. **Drop low-value columns** (`drop_low_value_columns`) - Automatically detects and drops ID-like columns (monotonic integers, names matching `id`, `index`, etc.) and free-text columns (high cardinality + long strings). Before dropping, extracts useful numeric info (e.g., "3 BHK Flat" → `Title_extracted_num = 3`). Keeps text-named columns if they have low cardinality (<=50 unique, treated as categorical).

5. **Convert numeric strings** (`_try_convert_to_numeric`) - Auto-detects object columns with mostly numeric-like strings (e.g., "75.9 Lac", "1390 sqft") and converts them to actual numbers before imputation.

6. **Impute missing values** (`impute_missing`) - Configurable via `preprocessing.imputer` in `program.md`. Critical step — cannot be skipped.
   - `median` (default): fillna with column median for numerical, mode for categorical
   - `mean`: fillna with column mean for numerical, mode for categorical
   - `knn`: KNNImputer (k=5) for numerical, mode for categorical
   - `iterative`: IterativeImputer/MICE for numerical, mode for categorical

7. **Encode categoricals** (`encode_categoricals`) - Configurable via `preprocessing.encoder` in `program.md`. Critical step — cannot be skipped.
   - `label_onehot` (default): label encoding for binary, one-hot for low-cardinality (<=20 unique), label encoding for high-cardinality (>20 unique to avoid memory explosion)
   - `ordinal`: OrdinalEncoder for all categorical features
   - `target`: target/mean encoding — maps each category to the mean target value

8. **Normalize** (`normalize`) - Configurable via `preprocessing.scaler` in `program.md`. Critical step — cannot be skipped.
   - `standard` (default): StandardScaler
   - `minmax`: MinMaxScaler
   - `robust`: RobustScaler
   - `maxabs`: MaxAbsScaler

9. **Handle imbalance** (`handle_imbalance`) - Classification only. If minority/majority ratio < 0.8, applies SMOTE. Critical step — cannot be skipped.

10. **Feature selection** (`select_features`) - Only triggers if feature count > 20. Uses `SelectKBest` with `f_classif` or `f_regression`, keeping top 20. Skippable via `skip_steps: [feature_selection]`.

11. **Split data** (`split_data`) - 80/10/10 train/val/test split. Stratified for classification. Fixed `random_state=42`.

12. **Save** (`save_processed`) - Saves everything to `processed/data_splits.pkl`.

**Fallback chains:** If the configured method fails, the pipeline automatically tries the next alternative instead of crashing:
- Imputation: requested → `knn` → `iterative` → `mean` → `median`
- Encoding: requested → `label_onehot` → `ordinal` → `target`
- Normalization: requested → `standard` → `minmax` → `robust` → `maxabs`
- A warning is printed when a fallback is used. Only raises `RuntimeError` if all methods fail.

**Safety verification** (`verify_no_remaining_objects`): After encoding and at the end, raises `RuntimeError` if any non-numeric columns survive — guaranteeing `train.py` receives clean data.

---

### src/train.py - The Agent's Playground

**Purpose:** Defines what model to train and how. This is the only file the AI agent modifies.

**Access:** READ + WRITE for the agent. This is the only file the agent is allowed to change.

**Structure:**
- `load_data()` - Loads `processed/data_splits.pkl`. Never needs modification.
- `get_model(task_type)` - Returns the sklearn model instance. **This is the main function the agent edits** to try different models and hyperparameters.
- `get_scoring(task_type, metric)` - Returns the sklearn scoring string. Rarely needs modification.
- `evaluate_model(model, X_val, y_val, task_type, metric_name)` - Evaluates on validation set. Returns `(primary_score, all_metrics_dict)`. The primary score respects the configured metric from `program.md`. All metrics are computed regardless:
  - **Regression:** RMSE, MAE, R2 adjusted (penalizes model complexity: `1 - (1 - R2) * (n-1) / (n-p-1)`)
  - **Classification:** F1 weighted, accuracy, precision weighted, recall weighted
- `train()` - The main training function. Handles the full flow:
  1. Loads fully preprocessed data directly (no feature engineering needed — data is already clean numeric)
  2. Gets model from `get_model()`
  3. Runs k-fold cross-validation
  4. Fits the model on the full training set
  5. Evaluates on validation set (computes all metrics)
  6. Logs everything to MLflow (params, primary metric, all evaluation metrics, model artifact)
  7. Prints a JSON result to stdout (last line) for `run_experiment.py` to parse

**No preprocessing in train.py:** Since `prepare.py` guarantees fully numeric, NaN-free data, `train.py` contains zero preprocessing logic. It simply loads the numpy arrays and trains.

**Communication with run_experiment.py:** The JSON printed to stdout on the last line contains: `run_id`, `model_name`, `metric_name`, `cv_mean`, `cv_std`, `val_score`, `training_time`, `notes`, and `all_metrics` (dict of all computed evaluation metrics).

**MLflow integration:** `train.py` receives MLflow context via environment variables set by `run_experiment.py`:
- `MLFLOW_TRACKING_URI` - Where to log
- `MLFLOW_RUN_ID` - The run ID to log under (so it logs to the same run the orchestrator created)
- `MLFLOW_EXPERIMENT_NAME` - Experiment name (used as fallback if no run ID)

**What the agent changes between experiments:**
- Imports (e.g., add `from sklearn.svm import SVC`)
- `get_model()` body (different model class, different hyperparameters)
- The `notes` string in `train()` (why this change was made)
- The docstring at the top (current experiment description)

---

### src/run_experiment.py - The Orchestrator

**Purpose:** Ties everything together. This is the script the user (or agent) runs to execute one experiment iteration.

**Usage:**
```bash
python src/run_experiment.py                    # Normal run
python src/run_experiment.py --force-prepare    # Re-run preprocessing
python src/run_experiment.py --experiment-name my_exp  # Override experiment name
```

**Complete execution flow:**

1. **Parse config** - Reads `program.md` YAML frontmatter. Extracts MLflow settings, scaler type, top N.

2. **Determine preprocessing script** - Always uses `src/prepare.py`. The preprocessing strategies (imputer, encoder, scaler) are configured in `program.md`.

3. **Run preprocessing** (`run_prepare`) - Runs `src/prepare.py` as a subprocess. Skips if `processed/data_splits.pkl` already exists (unless `--force-prepare`).

4. **Set up MLflow** - Sets tracking URI, creates/gets experiment, creates an MlflowClient.

5. **Start MLflow run and execute training** - Creates a parent MLflow run, then runs `src/train.py` as a subprocess with MLflow environment variables. Parses the JSON result from train.py's stdout (last line).

6. **Check improvement** (`load_best_scores` / `save_best_scores`) - Loads `best_scores.json`, compares current `val_score` to `best_val_score`:
   - If `val_score > best_val_score`: marks as improved, resets no-improvement counter
   - Otherwise: increments `consecutive_no_improvement` counter

7. **Git commit or revert:**
   - **Improved** (`git_commit_train`): Stages `src/train.py`, commits with message format: `Experiment: {ModelName} | {metric}={score}` plus timestamp and notes
   - **Not improved** (`git_revert_train`): Runs `git checkout -- src/train.py` to discard the agent's changes

8. **Register model** (`register_model_if_top_n`): Regardless of whether the experiment improved, checks if the score qualifies for the top N models in the MLflow registry:
   - If fewer than N models are registered, registers it
   - If N or more exist, only registers if this score beats the worst of the current top N
   - After registration, cleans up by deleting any versions beyond top N

9. **Save scores and print summary** - Updates `best_scores.json` and prints a formatted summary showing model name, CV scores, validation score, whether it improved, best score so far, total experiments, and no-improvement streak. If streak >= 3, prints a warning to consider switching scalers.

---

### best_scores.json - Experiment History Tracker

**Purpose:** Persistent file that tracks the best score, total experiments, and full history across all experiment runs.

**Structure:**
```json
{
  "best_val_score": 0.95,
  "best_run_id": "abc123",
  "best_model_name": "RandomForestClassifier",
  "best_metrics": {
    "val_f1_weighted": 0.95,
    "val_accuracy": 0.94,
    "val_precision_weighted": 0.95,
    "val_recall_weighted": 0.94
  },
  "total_experiments": 10,
  "consecutive_no_improvement": 2,
  "history": [
    {
      "run_id": "abc123",
      "model_name": "RandomForestClassifier",
      "val_score": 0.95,
      "cv_mean": 0.93,
      "all_metrics": {
        "val_f1_weighted": 0.95,
        "val_accuracy": 0.94,
        "val_precision_weighted": 0.95,
        "val_recall_weighted": 0.94
      },
      "timestamp": "2026-03-15T03:15:30"
    }
  ]
}
```

**Who writes it:** `run_experiment.py` after every experiment.

**Who reads it:** The AI agent reads this to understand what has been tried and what the current best is. `run_experiment.py` reads it to determine if a new experiment improved.

---

### src/select_model.py - Model Promotion CLI

**Purpose:** Lets the user view all registered models and promote one to "Production" stage in MLflow.

**What "Production" means in MLflow:** "Production" is just a label (called a "stage") that MLflow lets you assign to a model version in the registry. It does not actually deploy anything. MLflow has four stages: `None` (just registered), `Staging` (candidate), `Production` (the chosen model), and `Archived` (retired/old). When you run `python select_model.py --rank 1`, it simply changes the stage label from "None" to "Production" in MLflow's metadata. If you later build a serving pipeline, you can query MLflow for "give me the model in Production stage" and load it programmatically. But in AgentML, it's purely organizational — a way to mark which model you consider the best after all experiments are done.

**Commands:**
- `python src/select_model.py --list` - Displays a ranked table of all registered model versions with val_score, cv_mean, cv_std, and training time
- `python src/select_model.py --rank 1` - Promotes the #1 ranked model to Production stage, archiving any previous Production model

**How it works:**
1. Reads `program.md` to get MLflow settings
2. Constructs registry name as `agentml-{experiment_name}`
3. Fetches all model versions from MLflow registry
4. For each version, looks up the associated run to get metrics
5. Sorts by `val_score` descending
6. For `--rank`: transitions the selected version to "Production" stage and archives any existing Production version

---

### src/visualization/analysis.ipynb - Experiment Visualization Notebook

**Purpose:** Auto-generates visualizations by pulling data directly from MLflow.

**Configuration cell (first code cell):**
- `EXPERIMENT_NAME` - Which experiment to analyze
- `SPECIFIC_RUN_ID` - Analyze a single run (or `None` for all)
- `TRACKING_URI` - MLflow tracking location
- `TOP_N` - Number of top models to show in summary

**Setup cell:** Connects to MLflow, fetches all runs, builds a pandas DataFrame (`df_runs`) with columns for run_id, model_name, val_score, cv_mean, cv_std, training_time, agent_notes, start_time, experiment_num, and all hyperparameters (prefixed with `param_`). Also loads `processed/data_splits.pkl` for task type and data access.

**Visualizations generated:**

1. **Experiment Comparison** - Two plots side by side:
   - Bar chart of validation score per experiment
   - Line chart showing metric progression over time with CV std shaded region and model name annotations

2. **Feature Importance** - Finds the best tree-based model (RandomForest, GradientBoosting, XGBoost, ExtraTrees, DecisionTree), loads it from MLflow, and plots horizontal bar chart of `feature_importances_`. Shows top 20 features.

3. **Confusion Matrix / Residual Plot** - Loads the overall best model:
   - Classification: confusion matrix heatmap
   - Regression: predicted vs actual scatter plot + residual distribution histogram

4. **Learning Curve** - Loads the best model, runs `sklearn.model_selection.learning_curve` with 10 training size points and 5-fold CV. Shows training score vs cross-validation score as training size increases, with std shaded regions.

5. **CV Score Distribution** - Bar chart with error bars showing cv_mean +/- cv_std for every experiment, labeled with experiment number and model name.

6. **Summary Table** - Prints detailed info for top N models: rank, model name, experiment number, val_score, cv_mean, cv_std, training time, agent notes, and key hyperparameters. Also displays as a pandas DataFrame.

---

## Data Flow

### Complete flow of a single experiment:

```
1. User drops CSV into data/
2. User fills in program.md
3. User (or agent) runs: python src/run_experiment.py

   src/run_experiment.py:
   ├── Parses program.md (YAML frontmatter)
   ├── Runs src/prepare.py as subprocess
   │   ├── Loads CSV from data/
   │   ├── Runs preprocessing pipeline
   │   └── Saves processed/data_splits.pkl
   ├── Sets up MLflow (tracking URI, experiment)
   ├── Creates MLflow parent run
   ├── Runs src/train.py as subprocess (passes MLflow env vars)
   │   ├── Loads processed/data_splits.pkl
   │   ├── Creates model via get_model()
   │   ├── Runs k-fold cross-validation
   │   ├── Fits model on full training set
   │   ├── Evaluates on validation set
   │   ├── Logs params + metrics + model to MLflow
   │   └── Prints JSON result to stdout
   ├── Parses JSON result from train.py stdout
   ├── Loads best_scores.json
   ├── Compares val_score to best_val_score
   ├── If improved:
   │   ├── git add src/train.py && git commit
   │   └── Updates best_val_score in best_scores.json
   ├── If not improved:
   │   ├── git checkout -- src/train.py (revert)
   │   └── Increments consecutive_no_improvement
   ├── Registers model in MLflow registry if it qualifies for top N
   ├── Saves updated best_scores.json
   └── Prints experiment summary
```

### The agent loop (multiple experiments):

```
Agent reads: program.md, best_scores.json, src/train.py
    │
    ├── Decides what to try next
    ├── Edits src/train.py (get_model function, imports, notes)
    ├── Runs: python src/run_experiment.py
    ├── Reads output to see if improved
    │   ├── YES → change was committed, build on it
    │   └── NO  → change was reverted, try something different
    │
    └── Repeats
```

---

## Inter-Process Communication

The files communicate through several mechanisms:

| From | To | Mechanism |
|------|----|-----------|
| `program.md` | All scripts | YAML frontmatter parsed by each script |
| `src/run_experiment.py` | `src/prepare.py` | Subprocess call |
| `src/run_experiment.py` | `src/train.py` | Subprocess call + environment variables (`MLFLOW_TRACKING_URI`, `MLFLOW_RUN_ID`, `MLFLOW_EXPERIMENT_NAME`) |
| `src/train.py` | `src/run_experiment.py` | JSON printed to stdout (last line) |
| `src/prepare.py` | `src/train.py` | `processed/data_splits.pkl` file (pickle) |
| `src/run_experiment.py` | `src/run_experiment.py` (next run) | `best_scores.json` file |
| `src/train.py` | `src/visualization/analysis.ipynb` | MLflow tracking data (`mlruns/`) |
| `src/train.py` | `src/select_model.py` | MLflow model registry (`mlruns/`) |

---

## Key Design Decisions

1. **Subprocess isolation:** `src/prepare.py` and `src/train.py` run as subprocesses, not imports. This means crashes in training don't kill the orchestrator, and the agent's modifications to `src/train.py` are always loaded fresh.

2. **JSON stdout protocol:** `src/train.py` communicates results to `src/run_experiment.py` by printing a JSON object on the last line of stdout. All other print statements from train.py are captured and displayed. This is simple and reliable.

3. **Git as a local undo/redo mechanism:** Git is used purely locally here — nothing is ever pushed to GitHub or any remote. When an experiment improves, `src/run_experiment.py` runs `git add src/train.py` followed by `git commit` to save the change locally. When an experiment doesn't improve, it runs `git checkout -- src/train.py` to discard the agent's changes and restore `src/train.py` to the last committed (i.e., last successful) version. This means `git log` becomes a history of every successful improvement, and the agent always starts its next attempt from the last known good state. Pushing to a remote is never done automatically — that's entirely up to the user.

4. **MLflow for everything else:** Metrics, hyperparameters, model artifacts, and the model registry all live in MLflow. This gives full experiment tracking even for reverted experiments (the MLflow run still exists, just the train.py change was reverted).

5. **Single file modification:** Constraining the agent to only modify `src/train.py` dramatically simplifies the system. The agent can't break preprocessing, can't break orchestration, and every experiment is a clean diff on one file.

6. **Preprocessed data caching:** `processed/data_splits.pkl` is created once and reused across all experiments. Use `--force-prepare` to regenerate. This avoids re-preprocessing on every experiment run.

---

## File Access Permissions Summary

| File | Agent Can Read | Agent Can Write | Notes |
|------|:-:|:-:|-------|
| `src/train.py` | Yes | **Yes** | The only writable file |
| `program.md` | Yes | Limited | Can change `scaler` field only |
| `src/prepare.py` | Yes | No | READ ONLY - never modify |
| `src/run_experiment.py` | Yes | No | Orchestrator |
| `src/select_model.py` | Yes | No | CLI tool |
| `src/visualization/analysis.ipynb` | Yes | No | Visualization |
| `best_scores.json` | Yes | No | Written by orchestrator |
| `processed/data_splits.pkl` | Yes | No | Written by prepare scripts |
| `mlruns/` | Yes | No | Written by MLflow |

---

## Generated/Runtime Files

These files are created at runtime and are not part of the source code:

| File/Directory | Created By | Purpose |
|---|---|---|
| `data/*.csv` | User | Raw dataset |
| `processed/data_splits.pkl` | `src/prepare.py` | Preprocessed and split data |
| `best_scores.json` | `src/run_experiment.py` | Tracks best score, experiment history |
| `mlruns/` | MLflow | All experiment tracking data, model artifacts, registry |
| `.venv/` | `uv venv` | Python virtual environment |
