import json
import os
from pathlib import Path

import pandas as pd

from pipeline.promote import read_current_version
from pipeline.run import run_pipeline


def write_csv(path: Path, rows: list[list[int | float]]):
    df = pd.DataFrame(
        rows,
        columns=[
            "age",
            "income",
            "account_balance",
            "transactions_last_30d",
            "is_premium",
            "target",
        ],
    )
    df.to_csv(path, index=False)


def read_prod_score(artifacts_dir: Path) -> float:
    current_version = read_current_version(artifacts_dir / "models")
    assert current_version is not None
    meta_path = artifacts_dir / "models" / "versions" / current_version / "metadata.json"
    meta = json.loads(meta_path.read_text())
    return float(meta["metric"]["value"])


def test_pipeline_promotes_then_rejects(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))

    # Day A - decent signal - should promote (no prior model)
    csv_a = tmp_path / "day_a.csv"
    write_csv(
        csv_a,
        [
            [25, 52000, 1200, 14, 0, 0],
            [42, 88000, 5400, 33, 1, 1],
            [31, 61000, 2300, 18, 0, 0],
            [58, 99000, 15000, 41, 1, 1],
            [37, 72000, 3100, 25, 0, 0],
            [52, 91000, 12000, 38, 1, 1],
            [34, 65000, 2700, 21, 0, 0],
            [60, 105000, 18000, 44, 1, 1],
        ],
    )
    run_pipeline(csv_a)

    current_a = read_current_version(artifacts_dir / "models")
    assert current_a is not None
    prod_model = artifacts_dir / "models" / "versions" / current_a / "model.joblib"
    prod_meta = artifacts_dir / "models" / "versions" / current_a / "metadata.json"
    assert prod_model.exists()
    assert prod_meta.exists()

    score_a = read_prod_score(artifacts_dir)

    # Day B - intentionally awful / noisy - should be rejected (score likely 0)
    csv_b = tmp_path / "day_b.csv"
    write_csv(
        csv_b,
        [
            [25, 52000, 1200, 14, 0, 1],
            [42, 88000, 5400, 33, 1, 0],
            [31, 61000, 2300, 18, 0, 1],
            [58, 99000, 15000, 41, 1, 0],
            [37, 72000, 3100, 25, 0, 1],
            [52, 91000, 12000, 38, 1, 0],
            [34, 65000, 2700, 21, 0, 1],
            [60, 105000, 18000, 44, 1, 0],
        ],
    )
    run_pipeline(csv_b)

    # Production score should not decrease because reject keeps current version.
    score_after = read_prod_score(artifacts_dir)
    assert score_after == score_a
    assert read_current_version(artifacts_dir / "models") == current_a

    # Both versions are retained for rollback.
    versions = list((artifacts_dir / "models" / "versions").iterdir())
    assert len(versions) == 2
