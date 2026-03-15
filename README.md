# AgentML

Autonomous ML experimentation loop inspired by [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch). An AI agent iterates on model selection and hyperparameter tuning overnight by modifying a single file (`train.py`), running experiments, checking if the metric improved, committing the change via git if it did, reverting if it did not, and repeating the loop.

All experiment tracking is handled by MLflow. All dependency management uses uv.

## Project Structure

```
AgentML/
├── data/                             # Drop your CSV dataset here
├── processed/                        # Processed data splits (auto-generated)
├── prepare.py                        # Data preprocessing pipeline (READ ONLY)
├── prepare_if_only_when_required.py  # Alternative preprocessing with extra scalers (READ ONLY)
├── train.py                          # Training script (ONLY file the agent modifies)
├── program.md                        # Configuration template (fill this in)
├── run_experiment.py                 # Orchestration script
├── select_model.py                   # CLI tool to promote models
├── analysis.ipynb                    # Result visualization notebook
├── pyproject.toml                    # Dependencies (managed by uv)
└── README.md                         # This file
```

## Setup

### 1. Install uv

If you don't have uv installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and enter the repo

```bash
git clone <your-repo-url>
cd AgentML
```

### 3. Create a virtual environment

```bash
uv venv
```

### 4. Activate the virtual environment

On Mac/Linux:
```bash
source .venv/bin/activate
```

On Windows:
```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
uv sync
```

That's it. The environment is ready.

### Adding new dependencies

Instead of `pip install`, always use:

```bash
uv add package-name
```

This automatically updates `pyproject.toml` as well.

## Usage

### Step 1: Prepare your data

Drop your CSV dataset into the `data/` folder.

### Step 2: Configure program.md

Open `program.md` and fill in:

- **Dataset**: CSV filename, target column name, task type
- **Metric**: Which metric to optimize (or `auto` for automatic detection)
- **Model constraints**: What models the agent should explore
- **Preprocessing overrides**: Any preprocessing steps to skip
- **MLflow settings**: Experiment name, number of top models to track

### Step 3: Run the experiment

```bash
python run_experiment.py
```

This will:
1. Preprocess the data (if not already done)
2. Run `train.py` with the current model configuration
3. Log everything to MLflow
4. Commit `train.py` if the model improved, revert if it didn't
5. Print a summary

To force re-preprocessing:
```bash
python run_experiment.py --force-prepare
```

### Step 4: Let the agent iterate

The AI agent (Claude Code or similar) reads the experiment results, modifies `train.py` with a new model or hyperparameters, and runs `python run_experiment.py` again. This loop repeats until the agent is satisfied or the user stops it.

### Step 5: View and promote models

List all registered models:
```bash
python select_model.py --list
```

Promote the top model to production:
```bash
python select_model.py --rank 1
```

### Step 6: Analyze results

Open `analysis.ipynb` in Jupyter and set the experiment name in the first config cell. The notebook auto-generates:
- Experiment comparison charts
- Feature importance plots
- Confusion matrix or residual plots
- Learning curves
- CV score distributions
- Summary tables

## File Access Rules

These rules are critical for the agent to follow:

| File | Agent Access |
|------|-------------|
| `train.py` | **READ + WRITE** — The only file the agent can modify |
| `prepare.py` | **READ ONLY** — Never modify |
| `prepare_if_only_when_required.py` | **READ ONLY** — Agent can switch to it by changing `scaler` in `program.md`, but cannot modify the file |
| `program.md` | **READ** — Agent reads config; can update `scaler` field if StandardScaler underperforms |
| All other files | **READ ONLY** |

## How the Agent Loop Works

```
1. Agent reads program.md for task config and constraints
2. Agent reads MLflow results from previous experiments
3. Agent reads current train.py
4. Agent decides next experiment (new model, hyperparameters, ensemble, etc.)
5. Agent edits train.py
6. Agent runs: python run_experiment.py
7. Orchestrator checks if metric improved:
   - Improved → git commit train.py, register model in MLflow
   - Not improved → git revert train.py
8. Agent reads results, repeats from step 2
```

If the validation metric hasn't improved for 3 consecutive experiments, the agent should consider switching to `prepare_if_only_when_required.py` by changing the `scaler` field in `program.md` from `standard` to `minmax`, `robust`, or `maxabs`.

## program.md Examples

### Go wild (full freedom)

```yaml
dataset:
  path: data/customer_churn.csv
  target_column: churned
  task_type: classification

constraints:
  cv_folds: 10
```

```markdown
# Model Constraints
**Instructions:** go wild
```

### Only tree-based models

```yaml
dataset:
  path: data/housing.csv
  target_column: price
  task_type: regression
```

```markdown
# Model Constraints
**Instructions:** only use tree based models (RandomForest, GradientBoosting, XGBoost, ExtraTrees)
```

### Linear models first, then ensembles

```markdown
# Model Constraints
**Instructions:** try all linear models first (LogisticRegression, Ridge, Lasso, ElasticNet, SVM with linear kernel). Once linear models plateau, move to ensemble methods (VotingClassifier, StackingClassifier).

# Experiment Instructions
**Instructions:** focus on reducing overfitting. Use strong regularization.
```

### Skip preprocessing steps

```yaml
preprocessing:
  skip_steps: [normalization, smote]
  scaler: standard
```

## Key Constraints

- Every experiment uses 10-fold cross validation (configurable in `program.md`)
- Git tracks every change to `train.py` with descriptive commit messages
- Failed experiments (no improvement) are reverted via git
- MLflow logs all metrics, hyperparameters, and model artifacts
- Top N models are registered in the MLflow model registry
