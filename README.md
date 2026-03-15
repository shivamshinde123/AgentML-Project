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

You can use the following prompt directly with your AI agent to kick off the autonomous loop:

```
You are an autonomous ML experimentation agent. Your job is to iterate on
train.py to find the best model and hyperparameters for the task defined in
program.md.

Rules:
- train.py is the ONLY file you are allowed to modify.
- NEVER modify prepare.py, prepare_if_only_when_required.py, or run_experiment.py.
- Read program.md to understand the dataset, task type, metric, and any
  model constraints the user has specified.
- Read best_scores.json (if it exists) to see what has been tried so far and
  what the current best validation score is.
- If you need to install any new package, use `uv add package-name`. NEVER
  use pip.

Workflow — repeat this loop:
1. Read program.md, best_scores.json, and the current train.py.
2. Decide what to try next: a different model, different hyperparameters,
   regularization, or an ensemble (VotingClassifier, StackingClassifier).
3. Edit train.py with your changes. Update the get_model() function, the
   imports, and the notes string to describe why you made this change.
4. Run: python run_experiment.py
5. Read the output. The orchestrator will tell you if the metric improved.
   - If improved: your change was committed automatically. Build on it.
   - If not improved: your change was reverted automatically. Try something
     different next time.
6. Repeat from step 1.

Strategy guidance:
- Start with simple models (LogisticRegression, SVC) before trying complex
  ones (XGBoost, ensembles).
- Tune hyperparameters systematically — don't just guess randomly.
- If 3+ consecutive experiments show no improvement, consider changing the
  scaler in program.md (from standard to minmax, robust, or maxabs) which
  will trigger prepare_if_only_when_required.py.
- Always use cross-validation scores to judge generalization, not just the
  validation score.
- Log clear notes explaining your reasoning for each experiment.

Keep iterating until you have run at least 10 experiments or the validation
score has plateaued for 5+ consecutive attempts.
```

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
