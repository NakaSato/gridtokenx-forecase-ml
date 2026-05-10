import os, sys, pickle, subprocess, json, threading
import numpy as np
from typing import List, Optional
from domain.entities import TelemetryRow
from domain.interfaces import IPredictor

# We keep the lazy-loading logic but encapsulate it
class HybridPredictor(IPredictor):
    def __init__(self, root_dir: str, device: str = "cpu"):
        self.root_dir = root_dir
        self.device = device
        self._torch = None
        self._net = None
        self._meta = None
        self._tc = {"window_size": 48, "forecast_horizon": 24}
        self._seq_fields = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._net is not None:
            return
        
        with self._lock:
            if self._net is not None:
                return
            
            import torch
            self._torch = torch
            
            # Import models dynamically to avoid circular dependencies or early torch/lgbm init
            from models import tcn_model
            from models.device import get_device
            
            self.device = get_device()
            
            try:
                ckpt_path = os.path.join(self.root_dir, "models/tcn.pt")
                CKPT = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                self._tc = CKPT["config"]
                self._net = tcn_model.TCN(
                    CKPT["in_features"], self._tc["filters"], self._tc["kernel_size"],
                    self._tc["layers"], self._tc["forecast_horizon"],
                    dropout=CKPT.get("dropout", 0.0),
                ).to(self.device)
                self._net.load_state_dict(CKPT["state_dict"])
                self._net.eval()
                self._seq_fields = [f.lower() for f in tcn_model.SEQ_FEATURES]
            except Exception as e:
                print(f"⚠️  TCN model load failed: {e}")

            try:
                meta_path = os.path.join(self.root_dir, "models/meta_learner.pkl")
                with open(meta_path, "rb") as f:
                    self._meta = pickle.load(f)
            except Exception as e:
                print(f"⚠️  Meta-learner load failed: {e}")

    def _lgbm_predict(self, features: dict) -> List[float]:
        """Run LightGBM prediction in a clean subprocess."""
        script = f"""
import pickle, os, sys, json
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = {repr(self.root_dir)}
sys.path.insert(0, ROOT)
with open(os.path.join(ROOT, "models/lgbm.pkl"), "rb") as f:
    model = pickle.load(f)
features = json.loads({repr(json.dumps(features))})
from models.lgbm_model import FEATURES
X_row = [features.get(c, 0.0) for c in FEATURES]
X = np.array([X_row])
preds = model.predict(X)[0].tolist()
print(json.dumps(preds))
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, cwd=self.root_dir, timeout=10
            )
            if result.returncode != 0:
                return [0.0] * 3 # Default 3 targets
            return json.loads(result.stdout.strip())
        except Exception:
            return [0.0] * 3

    def _tcn_predict(self, history: List[TelemetryRow]) -> np.ndarray:
        if not self._net:
            return np.zeros((self._tc["forecast_horizon"], 3))
        
        arr = np.array([[getattr(r, f) for f in self._seq_fields] for r in history], dtype=np.float32)
        x = self._torch.tensor(arr).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            return self._net(x).cpu().numpy()[0]

    def predict(self, history: List[TelemetryRow], lgbm_features: dict) -> List[float]:
        self._ensure_loaded()
        
        if not self._meta:
            return [0.0] * self._tc["forecast_horizon"]
        
        horizon = self._tc["forecast_horizon"]
        lgbm_preds = self._lgbm_predict(lgbm_features)
        tcn_preds = self._tcn_predict(history)
        
        # Target 0: tao_load_mw
        X_meta = np.column_stack([np.full(horizon, lgbm_preds[0]), tcn_preds[:, 0]])
        return self._meta[0].predict(X_meta).tolist()

    @property
    def window_size(self):
        self._ensure_loaded()
        return self._tc["window_size"]

    @property
    def seq_fields(self):
        self._ensure_loaded()
        return self._seq_fields
