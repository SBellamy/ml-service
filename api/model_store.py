import json
import os
import threading
from pathlib import Path
import joblib

class ModelStore:
    def __init__(self, artifacts_dir: str, model_subdir: str | None):
        self._lock = threading.RLock()
        self.artifacts_dir = Path(artifacts_dir)
        self.model_subdir = model_subdir
        self.model = None
        self.metadata = None
        self._loaded_ref: str | None = None

    def _resolve_model_dir(self) -> tuple[Path | None, str | None]:
        if self.model_subdir:
            return self.artifacts_dir / self.model_subdir, f"subdir:{self.model_subdir}"

        models_root = self.artifacts_dir / "models"
        pointer = models_root / "CURRENT"
        if pointer.exists():
            version = pointer.read_text().strip()
            if version:
                return models_root / "versions" / version, f"version:{version}"

        # backwards compatability from before model versioning was implemented
        legacy = models_root / "production"
        if legacy.exists():
            return legacy, "legacy:production"
        return None, None

    def _refresh_locked(self, force: bool = False) -> None:
        model_dir, model_ref = self._resolve_model_dir()
        if model_dir is None or model_ref is None:
            self.model = None
            self.metadata = None
            self._loaded_ref = None
            return

        if not force and self.model is not None and self._loaded_ref == model_ref:
            return

        model_path = model_dir / "model.joblib"
        meta_path = model_dir / "metadata.json"
        if not model_path.exists() or not meta_path.exists():
            self.model = None
            self.metadata = None
            self._loaded_ref = None
            return

        self.model = joblib.load(model_path)
        self.metadata = json.loads(meta_path.read_text())
        self._loaded_ref = model_ref

    def load(self):
        with self._lock:
            self._refresh_locked(force=True)

    def ready(self):
        with self._lock:
            self._refresh_locked(force=False)
            return self.model is not None

    def predict(self, features):
        with self._lock:
            self._refresh_locked(force=False)
            if self.model is None:
                raise RuntimeError("Model not ready")
            proba = self.model.predict_proba([features])[0][1]
            pred = int(proba >= 0.5)
            return pred, float(proba)

def build_store():
    artifacts = os.environ.get("ARTIFACTS_DIR", "/artifacts")
    subdir = os.environ.get("MODEL_SUBDIR")
    return ModelStore(artifacts, subdir)
