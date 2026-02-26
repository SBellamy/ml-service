import argparse
import json
import os
import time
from pathlib import Path
import pandas as pd
import joblib

from pipeline.train import train_model
from pipeline.validate import validate_df
from pipeline.evaluate import evaluate
from pipeline.metadata import build_model_metadata
from pipeline.promote import (
    new_version_id,
    promote_version_atomically,
    read_current_score,
    validate_model_artifacts,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    return parser.parse_args()

def run_pipeline(csv_path: Path):
    df = pd.read_csv(csv_path)

    artifacts = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    models_root = artifacts / "models"
    versions_dir = models_root / "versions"
    
    validate_df(df)
    print("Data validation passed!")

    model, X_test, y_test = train_model(df)
    print("Model training complete!")

    score = evaluate(model, X_test, y_test)
    print("Model evaluation complete!")

    version = new_version_id()
    version_dir = versions_dir / version
    version_dir.mkdir(parents=True, exist_ok=False)

    meta = build_model_metadata(
        model_version_id=version,
        metric_name="f1",
        metric_value=score,
        training_data_file=csv_path,
        trained_at=int(time.time()),
    )

    joblib.dump(model, version_dir / "model.joblib")
    (version_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    validate_model_artifacts(version_dir)

    current_score = read_current_score(models_root)
    promote = current_score is None or score > current_score

    if promote:
        promote_version_atomically(models_root, version)
        decision = "PROMOTED"
    else:
        decision = "REJECTED"

    print(json.dumps({"decision": decision, "score": score, "version": version}, indent=2))

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(Path(args.csv))
