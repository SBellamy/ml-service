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

    def _resolve_model_dir(self) -> Path | None:
        if self.model_subdir:
            return self.artifacts_dir / self.model_subdir

        models_root = self.artifacts_dir / "models"
        pointer = models_root / "CURRENT"
        if pointer.exists():
            version = pointer.read_text().strip()
            if version:
                return models_root / "versions" / version

        # backwards compatability from before model versioning was implemented
        legacy = models_root / "production"
        if legacy.exists():
            return legacy
        return None

    def load(self):
        with self._lock:
            model_dir = self._resolve_model_dir()
            if model_dir is None:
                self.model = None
                self.metadata = None
                return

            model_path = model_dir / "model.joblib"
            meta_path = model_dir / "metadata.json"
            if not model_path.exists() or not meta_path.exists():
                self.model = None
                self.metadata = None
                return
            self.model = joblib.load(model_path)
            self.metadata = json.loads(meta_path.read_text())

    def ready(self):
        with self._lock:
            return self.model is not None

    def predict(self, features):
        with self._lock:
            proba = self.model.predict_proba([features])[0][1]
            pred = int(proba >= 0.5)
            return pred, float(proba)

def build_store():
    artifacts = os.environ.get("ARTIFACTS_DIR", "/artifacts")
    subdir = os.environ.get("MODEL_SUBDIR")
    return ModelStore(artifacts, subdir)
