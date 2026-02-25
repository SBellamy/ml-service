import argparse
import json
import os
import time
from pathlib import Path
import shutil
import pandas as pd
import joblib
from sklearn.metrics import classification_report

from pipeline.train import train_model
from pipeline.validate import validate_df
from pipeline.evaluate import evaluate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    return parser.parse_args()

def run_pipeline(csv_path: Path):
    df = pd.read_csv(csv_path)

    artifacts = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    staging = artifacts / "models/staging"
    production = artifacts / "models/production"
    
    validate_df(df)
    print("Data validation passed!")

    model, X_test, y_test = train_model(df)
    print("Model training complete!")

    score = evaluate(model, X_test, y_test)
    print("Model evaluation complete!")

    meta = {
        "metric_name": "f1",
        "score": score,
        "trained_at": int(time.time()),
    }

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, staging / "model.joblib")
    (staging / "metadata.json").write_text(json.dumps(meta, indent=2))

    promote = True
    if (production / "metadata.json").exists():
        current = json.loads((production / "metadata.json").read_text())
        promote = score > current["score"]

    if promote:
        if production.exists():
            shutil.rmtree(production)
        shutil.move(staging, production)
        decision = "PROMOTED"
    else:
        decision = "REJECTED"

    print(json.dumps({"decision": decision, "score": score}, indent=2))

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(Path(args.csv))
