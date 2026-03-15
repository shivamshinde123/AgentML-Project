"""
AgentML Model Selector

CLI tool to view and promote models from the MLflow registry.

Usage:
    python select_model.py --list                   # List all registered models
    python select_model.py --rank 1                  # Promote top-ranked model to Production
    python select_model.py --rank 1 --experiment-name my_exp  # Specify experiment
"""

import argparse
import yaml

import mlflow
from mlflow.tracking import MlflowClient


def parse_program_md(path="program.md"):
    """Parse YAML frontmatter from program.md."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = content.split("---", 2)
    if len(parts) >= 3:
        config = yaml.safe_load(parts[1])
    else:
        config = {}
    return config


def get_registered_models(client, registry_name):
    """Get all registered model versions with their metrics, sorted by val_score."""
    try:
        versions = client.search_model_versions(f"name='{registry_name}'")
    except mlflow.exceptions.MlflowException:
        print(f"No registered model found with name '{registry_name}'")
        return []

    model_info = []
    for v in versions:
        try:
            run = client.get_run(v.run_id)
            metrics = run.data.metrics
            params = run.data.params
            model_info.append({
                "version": v.version,
                "run_id": v.run_id,
                "model_name": params.get("model_name", "unknown"),
                "val_score": metrics.get("val_score", float("-inf")),
                "cv_mean": metrics.get("cv_mean", float("-inf")),
                "cv_std": metrics.get("cv_std", 0),
                "training_time": metrics.get("training_time", 0),
                "status": v.status,
                "current_stage": v.current_stage if hasattr(v, "current_stage") else "None",
                "description": v.description or "",
            })
        except Exception as e:
            model_info.append({
                "version": v.version,
                "run_id": v.run_id,
                "model_name": "error",
                "val_score": float("-inf"),
                "cv_mean": float("-inf"),
                "cv_std": 0,
                "training_time": 0,
                "status": v.status,
                "current_stage": "None",
                "description": str(e),
            })

    # Sort by val_score descending
    model_info.sort(key=lambda x: x["val_score"], reverse=True)
    return model_info


def list_models(client, registry_name):
    """Print a table of all registered models."""
    models = get_registered_models(client, registry_name)

    if not models:
        print("No models registered yet. Run some experiments first!")
        return

    print(f"\nRegistered Models: {registry_name}")
    print("=" * 100)
    print(f"{'Rank':<6}{'Version':<9}{'Model':<30}{'Val Score':<14}"
          f"{'CV Mean':<14}{'CV Std':<12}{'Time(s)':<10}")
    print("-" * 100)

    for i, m in enumerate(models, 1):
        print(f"{i:<6}{m['version']:<9}{m['model_name']:<30}"
              f"{m['val_score']:<14.6f}{m['cv_mean']:<14.6f}"
              f"{m['cv_std']:<12.6f}{m['training_time']:<10.2f}")

    print("=" * 100)
    print(f"Total: {len(models)} model(s)")


def promote_model(client, registry_name, rank):
    """Promote the model at the given rank to Production stage."""
    models = get_registered_models(client, registry_name)

    if not models:
        print("No models registered yet.")
        return

    if rank < 1 or rank > len(models):
        print(f"Invalid rank {rank}. Available ranks: 1 to {len(models)}")
        return

    selected = models[rank - 1]
    version = selected["version"]
    model_name = selected["model_name"]
    val_score = selected["val_score"]

    # Transition to Production (archive any existing Production model)
    try:
        # First, check for existing Production versions and archive them
        for m in models:
            if m.get("current_stage") == "Production" and m["version"] != version:
                client.transition_model_version_stage(
                    name=registry_name,
                    version=m["version"],
                    stage="Archived",
                )
                print(f"Archived previous production model (version {m['version']})")

        client.transition_model_version_stage(
            name=registry_name,
            version=version,
            stage="Production",
        )
        print(f"\nPromoted to Production:")
        print(f"  Model:     {model_name}")
        print(f"  Version:   {version}")
        print(f"  Val Score:  {val_score:.6f}")
        print(f"  Run ID:    {selected['run_id']}")

    except Exception as e:
        print(f"Error promoting model: {e}")


def main():
    parser = argparse.ArgumentParser(description="AgentML Model Selector")
    parser.add_argument("--list", action="store_true",
                        help="List all registered models with metrics")
    parser.add_argument("--rank", type=int, default=None,
                        help="Promote the model at this rank to Production")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Override experiment name for registry lookup")
    parser.add_argument("--tracking-uri", type=str, default=None,
                        help="Override MLflow tracking URI")
    args = parser.parse_args()

    if not args.list and args.rank is None:
        parser.print_help()
        return

    # Get config
    config = parse_program_md()
    mlflow_config = config.get("mlflow", {})
    tracking_uri = args.tracking_uri or mlflow_config.get("tracking_uri", "./mlruns")
    experiment_name = args.experiment_name or mlflow_config.get(
        "experiment_name", "agentml_experiment"
    )
    registry_name = f"agentml-{experiment_name}"

    # Set up MLflow client
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    if args.list:
        list_models(client, registry_name)

    if args.rank is not None:
        promote_model(client, registry_name, args.rank)


if __name__ == "__main__":
    main()
