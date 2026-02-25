import json
import os
import threading
from pathlib import Path
import joblib

class ModelStore:
    def __init__(self, artifacts_dir: str, model_subdir: str):
        self._lock = threading.RLock()
        self.model_dir = Path(artifacts_dir) / model_subdir
        self.model = None
        self.metadata = None

    def load(self):
        with self._lock:
            model_path = self.model_dir / "model.joblib"
            meta_path = self.model_dir / "metadata.json"
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
    subdir = os.environ.get("MODEL_SUBDIR", "models/production")
    return ModelStore(artifacts, subdir)