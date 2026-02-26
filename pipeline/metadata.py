from pathlib import Path

MODEL_METADATA_SCHEMA_VERSION = "v1"
FEATURE_SCHEMA_VERSION = "v1"


def build_model_metadata(
    *,
    model_version_id: str,
    metric_name: str,
    metric_value: float,
    training_data_file: Path,
    trained_at: int,
) -> dict:
    return {
        "schema_version": MODEL_METADATA_SCHEMA_VERSION,
        "model_version_id": model_version_id,
        "metric": {
            "name": metric_name,
            "value": float(metric_value),
        },
        "training_data_file": training_data_file.name,
        "trained_at": trained_at,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
