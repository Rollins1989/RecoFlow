"""
MLflow tracking for the ranker: model, dataset version, feature version,
hyperparameters, ranking metrics, embedding model, candidate-generation
config — everything needed to compare ranker-v1/v2/v3 apples-to-apples and
to promote a model to the 'production' alias after approval.
"""
from __future__ import annotations

import os

import mlflow

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "recoflow-ranker"


def log_training_run(model, params: dict, metrics: dict, dataset_version: str,
                       feature_version: str, embedding_model: str,
                       candidate_generation_config: dict, model_name: str = "recoflow-ranker"):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "dataset_version": dataset_version,
            "feature_version": feature_version,
            "embedding_model": embedding_model,
        })
        mlflow.log_dict(candidate_generation_config, "candidate_generation_config.json")
        try:
            mlflow.lightgbm.log_model(model, artifact_path="model", registered_model_name=model_name)
        except Exception:
            mlflow.log_text(str(model), "model_repr.txt")
        return run.info.run_id


def promote_to_production(model_name: str, version: int):
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production",
        archive_existing_versions=True,
    )
