"""
AgentML Experiment Orchestrator

Ties everything together: reads program.md, runs preprocessing, executes
train.py, logs to MLflow, tracks best models, and handles git commit/revert.

Usage:
    python run_experiment.py
    python run_experiment.py --force-prepare
"""

import os
import sys
import json
import logging
import subprocess
import argparse
from datetime import datetime

# Resolve project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

import yaml
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def parse_program_md(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "program.md")
    """Parse YAML frontmatter from program.md."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        logger.error("Config file not found: %s", path)
        sys.exit(1)
    except IOError as e:
        logger.error("Could not read config file %s: %s", path, e)
        sys.exit(1)
    parts = content.split("---", 2)
    if len(parts) >= 3:
        try:
            config = yaml.safe_load(parts[1])
        except yaml.YAMLError as e:
            logger.error("Failed to parse YAML frontmatter: %s", e)
            sys.exit(1)
        instructions = parts[2].strip()
    else:
        config = {}
        instructions = content
    return config, instructions


def load_best_scores(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "best_scores.json")
    """Load the best scores tracking file."""
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning("Could not parse best_scores.json: %s. Starting fresh.", e)
        except IOError as e:
            logger.warning("Could not read best_scores.json: %s. Starting fresh.", e)
    return {
        "best_val_score": None,
        "best_run_id": None,
        "best_model_name": None,
        "total_experiments": 0,
        "consecutive_no_improvement": 0,
        "history": [],
    }


def save_best_scores(scores, path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "best_scores.json")
    """Save the best scores tracking file."""
    try:
        with open(path, "w") as f:
            json.dump(scores, f, indent=2)
    except IOError as e:
        logger.warning("Could not save best_scores.json: %s", e)


def run_prepare(config, force=False):
    """Run the preprocessing pipeline."""
    processed_path = os.path.join(PROJECT_ROOT, "processed", "data_splits.pkl")

    if os.path.exists(processed_path) and not force:
        logger.info("Processed data already exists, skipping preprocessing. "
                    "Use --force-prepare to re-run preprocessing.")
        return True

    script = os.path.join(SRC_DIR, "prepare.py")
    logger.info("Running preprocessing: %s", script)

    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True,
    )

    if result.stdout:
        logger.info(result.stdout.strip())
    if result.returncode != 0:
        logger.error("Preprocessing failed!\n%s", result.stderr)
        return False

    return True


def run_train(tracking_uri, run_id, experiment_name):
    """Run train.py as a subprocess with MLflow context."""
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = tracking_uri
    env["MLFLOW_RUN_ID"] = run_id
    env["MLFLOW_EXPERIMENT_NAME"] = experiment_name

    result = subprocess.run(
        [sys.executable, os.path.join(SRC_DIR, "train.py")],
        capture_output=True, text=True, env=env,
    )

    if result.returncode != 0:
        logger.error("Training failed!\n%s", result.stderr)
        return None

    # Parse JSON result from stdout (last line)
    stdout_lines = result.stdout.strip().split("\n")
    try:
        train_result = json.loads(stdout_lines[-1])
        # Log any other output from train.py (not the JSON)
        for line in stdout_lines[:-1]:
            logger.info(line)
        return train_result
    except (json.JSONDecodeError, IndexError):
        logger.warning("Could not parse train.py output")
        logger.info(result.stdout)
        return None


def git_commit_train(model_name, val_score, metric_name, notes=""):
    """Commit train.py changes with a descriptive message."""
    try:
        # Check if git is available and we're in a repo
        check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True,
        )
        if check.returncode != 0:
            logger.info("Not a git repo, skipping commit")
            return False

        # Stage train.py
        subprocess.run(["git", "add", "src/train.py"], capture_output=True, text=True)

        # Check if there are changes to commit
        diff = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True,
        )
        if not diff.stdout.strip():
            logger.info("No changes to train.py, skipping commit")
            return False

        # Commit with descriptive message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = (
            f"Experiment: {model_name} | {metric_name}={val_score:.6f}\n\n"
            f"Timestamp: {timestamp}\n"
            f"Notes: {notes}"
        )
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            capture_output=True, text=True,
        )
        logger.info("Committed train.py: %s (%s=%.6f)", model_name, metric_name, val_score)
        return True

    except FileNotFoundError:
        logger.info("Git not found, skipping commit")
        return False


def git_revert_train():
    """Revert train.py to the last committed version."""
    try:
        result = subprocess.run(
            ["git", "checkout", "--", "src/train.py"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info("Reverted train.py to last committed version")
            return True
        else:
            logger.warning("Could not revert train.py: %s", result.stderr)
            return False
    except FileNotFoundError:
        logger.info("Git not found, cannot revert")
        return False


def register_model_if_top_n(client, run_id, model_name, val_score,
                            experiment_name, top_n):
    """Register model in MLflow registry and keep only the top N versions."""
    registry_name = f"agentml-{experiment_name}"

    try:
        # Create registered model if it doesn't exist
        try:
            client.create_registered_model(registry_name)
        except mlflow.exceptions.MlflowException:
            pass  # Already exists

        # Get existing versions and their scores
        existing_versions = client.search_model_versions(f"name='{registry_name}'")
        version_scores = []
        for v in existing_versions:
            try:
                run = client.get_run(v.run_id)
                score = run.data.metrics.get("val_score", float("-inf"))
                version_scores.append((v, score))
            except Exception:
                version_scores.append((v, float("-inf")))

        # Check if this model qualifies for top N
        if len(version_scores) >= top_n:
            version_scores.sort(key=lambda x: x[1], reverse=True)
            worst_top_n_score = version_scores[top_n - 1][1]
            if val_score <= worst_top_n_score:
                logger.info("Model score %.6f does not qualify for top %d, "
                            "skipping registration", val_score, top_n)
                return False

        # Register this model version
        model_uri = f"runs:/{run_id}/model"
        mv = client.create_model_version(
            name=registry_name,
            source=model_uri,
            run_id=run_id,
            description=f"{model_name} | val_score={val_score:.6f}",
        )
        logger.info("Registered model version %s (%s, val_score=%.6f)",
                    mv.version, model_name, val_score)

        # Clean up: keep only top N versions
        all_versions = client.search_model_versions(f"name='{registry_name}'")
        if len(all_versions) > top_n:
            version_scores = []
            for v in all_versions:
                try:
                    run = client.get_run(v.run_id)
                    score = run.data.metrics.get("val_score", float("-inf"))
                    version_scores.append((v, score))
                except Exception:
                    version_scores.append((v, float("-inf")))

            version_scores.sort(key=lambda x: x[1], reverse=True)

            for v, score in version_scores[top_n:]:
                try:
                    client.delete_model_version(
                        name=registry_name, version=v.version
                    )
                    logger.info("Removed model version %s (outside top %d)",
                                v.version, top_n)
                except Exception:
                    pass

        return True

    except Exception as e:
        logger.warning("Could not register model: %s", e)
        return False


def print_summary(scores, train_result, improved):
    """Print experiment summary."""
    summary = (
        "\n" + "=" * 60 + "\n"
        "EXPERIMENT SUMMARY\n"
        + "=" * 60 + "\n"
        f"  Model:              {train_result['model_name']}\n"
        f"  CV Mean:            {train_result['cv_mean']:.6f} "
        f"(+/- {train_result.get('cv_std', 0):.6f})\n"
        f"  Validation Score:   {train_result['val_score']:.6f}\n"
        f"  Training Time:      {train_result['training_time']:.2f}s\n"
        f"  Improved:           {'YES' if improved else 'NO'}\n"
        f"  Best Score So Far:  {scores['best_val_score']:.6f}\n"
        f"  Best Model:         {scores['best_model_name']}\n"
        f"  Total Experiments:  {scores['total_experiments']}\n"
        f"  No-improvement streak: {scores['consecutive_no_improvement']}\n"
        + "=" * 60
    )
    logger.info(summary)
    if scores["consecutive_no_improvement"] >= 3:
        logger.warning("3+ consecutive experiments without improvement. "
                       "Consider changing imputer, encoder, or scaler in program.md.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="AgentML Experiment Orchestrator")
    parser.add_argument("--force-prepare", action="store_true",
                        help="Re-run preprocessing even if splits exist")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Override MLflow experiment name")
    args = parser.parse_args()

    # Parse config
    config, instructions = parse_program_md()
    mlflow_config = config.get("mlflow", {})
    raw_tracking_uri = mlflow_config.get("tracking_uri", "./mlruns")
    # Resolve relative tracking URIs against project root and use file: URI for MLflow
    if raw_tracking_uri.startswith(("http://", "https://", "databricks", "sqlite", "postgresql", "mysql", "mssql")):
        tracking_uri = raw_tracking_uri
    else:
        resolved = os.path.normpath(os.path.join(PROJECT_ROOT, raw_tracking_uri))
        tracking_uri = "file:///" + resolved.replace("\\", "/")
    experiment_name = args.experiment_name or mlflow_config.get(
        "experiment_name", "agentml_experiment"
    )
    top_n = mlflow_config.get("top_n_models", 5)

    logger.info("=" * 60)
    logger.info("AgentML Experiment Orchestrator")
    logger.info("  Experiment: %s", experiment_name)
    logger.info("  Tracking URI: %s", tracking_uri)
    logger.info("  Prepare script: src/prepare.py")
    logger.info("=" * 60)

    # Step 1: Run preprocessing
    if not run_prepare(config, force=args.force_prepare):
        logger.error("Preprocessing failed, aborting.")
        sys.exit(1)

    # Step 2: Set up MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Step 3: Start MLflow run and execute training
    with mlflow.start_run() as parent_run:
        run_id = parent_run.info.run_id
        logger.info("Started MLflow run: %s", run_id)

        # Run train.py
        train_result = run_train(tracking_uri, run_id, experiment_name)

        if train_result is None:
            logger.error("Training failed. Reverting train.py.")
            mlflow.set_tag("status", "FAILED")
            git_revert_train()
            sys.exit(1)

    # Step 4: Check if improved
    scores = load_best_scores()
    val_score = train_result["val_score"]
    model_name = train_result["model_name"]
    metric_name = train_result.get("metric_name", "val_score")
    notes = train_result.get("notes", "")

    scores["total_experiments"] += 1
    scores["history"].append({
        "run_id": train_result["run_id"],
        "model_name": model_name,
        "val_score": val_score,
        "cv_mean": train_result["cv_mean"],
        "all_metrics": train_result.get("all_metrics", {}),
        "timestamp": datetime.now().isoformat(),
    })

    improved = False
    if scores["best_val_score"] is None or val_score > scores["best_val_score"]:
        improved = True
        scores["best_val_score"] = val_score
        scores["best_run_id"] = train_result["run_id"]
        scores["best_model_name"] = model_name
        scores["best_metrics"] = train_result.get("all_metrics", {})
        scores["consecutive_no_improvement"] = 0
    else:
        scores["consecutive_no_improvement"] += 1

    # Step 5: Git commit or revert
    if improved:
        git_commit_train(model_name, val_score, metric_name, notes)
    else:
        logger.info("No improvement (%.6f <= %.6f). Reverting train.py.",
                    val_score, scores['best_val_score'])
        git_revert_train()

    # Step 5b: Register model if it's in the top N (regardless of improvement)
    register_model_if_top_n(
        client, train_result["run_id"], model_name,
        val_score, experiment_name, top_n,
    )

    # Step 6: Save scores and print summary
    save_best_scores(scores)
    print_summary(scores, train_result, improved)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        raise
