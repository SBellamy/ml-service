import json
import os
import time
from pathlib import Path


def validate_model_artifacts(model_dir: Path) -> None:
    model_path = model_dir / "model.joblib"
    meta_path = model_dir / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing model artifacts in {model_dir}")


def read_current_version(models_root: Path) -> str | None:
    pointer = models_root / "CURRENT"
    if not pointer.exists():
        return None
    version = pointer.read_text().strip()
    return version or None


def resolve_version_dir(models_root: Path, version: str) -> Path:
    return models_root / "versions" / version


def read_score(model_dir: Path) -> float:
    metadata = json.loads((model_dir / "metadata.json").read_text())
    return float(metadata["metric"]["value"])


def read_current_score(models_root: Path) -> float | None:
    current_version = read_current_version(models_root)
    if current_version is None:
        return None
    current_dir = resolve_version_dir(models_root, current_version)
    if not current_dir.exists():
        return None
    validate_model_artifacts(current_dir)
    return read_score(current_dir)


def new_version_id() -> str:
    return str(time.time_ns())


def promote_version_atomically(models_root: Path, version: str) -> None:
    models_root.mkdir(parents=True, exist_ok=True)
    version_dir = resolve_version_dir(models_root, version)
    validate_model_artifacts(version_dir)

    pointer = models_root / "CURRENT"
    pointer_tmp = models_root / f".CURRENT.tmp-{os.getpid()}-{time.time_ns()}"
    pointer_tmp.write_text(f"{version}\n")
    os.replace(pointer_tmp, pointer)
