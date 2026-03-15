---
dataset:
  path: data/iris.csv
  target_column: species
  task_type: auto                    # classification | regression | auto

metric:
  primary: auto                      # auto | f1_weighted | rmse | accuracy | r2 | neg_mean_squared_error

preprocessing:
  skip_steps: []                     # options: duplicates, imputation, encoding, normalization, smote, feature_selection
  scaler: standard                   # standard | minmax | robust | maxabs

mlflow:
  experiment_name: agentml_experiment
  tracking_uri: ./mlruns
  top_n_models: 5

constraints:
  cv_folds: 10
  max_training_time: 300             # seconds
---

# Model Constraints

Describe what models the agent should explore. Examples:

- "only use tree based models"
- "try all linear models first, then move to ensemble methods"
- "go wild" (agent has full freedom to try anything)

**Instructions:** go wild

# Preprocessing Overrides

List any preprocessing steps to skip. Leave empty if none.

# Experiment Instructions

Any additional free-form instructions for the agent. Examples:

- "focus on reducing overfitting"
- "prioritize inference speed over accuracy"
- "try stacking and voting ensembles after exhausting single models"

**Instructions:** None

# Analysis

Specify the experiment name or run ID to analyze in analysis.ipynb.

**Experiment name:** agentml_experiment
