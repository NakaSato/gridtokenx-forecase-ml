"""
Multi-Target Temporal Convolutional Network for sequential forecasting.
Targets: tao_load_mw, cable_flow_mw, kmb_flow_mw.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import mlflow
from models.schema import SEQ_FEATURES, TARGETS
from models.device import get_device
from models.mlflow_utils import setup_mlflow

setup_mlflow()
mlflow.set_experiment("GridTokenX_TCN")

# ── Feature Set ─────────────────────────────────
# Features and targets imported from schema.py


# ── Model ────────────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=self.pad)

    def forward(self, x):
        return self.conv(x)[:, :, :-self.pad] if self.pad else self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x) if self.downsample else x
        return self.dropout(torch.relu(out + res))


class TCN(nn.Module):
    def __init__(self, in_features, filters, kernel_size, n_layers, horizon, n_targets=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = in_features if i == 0 else filters
            layers.append(TCNBlock(in_ch, filters, kernel_size, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc  = nn.Linear(filters, horizon * n_targets)
        self.n_targets = n_targets
        self.horizon = horizon

    def forward(self, x):          # x: (B, T, F)
        x = x.permute(0, 2, 1)    # → (B, F, T)
        x = self.tcn(x)            # → (B, filters, T)
        out = self.fc(x[:, :, -1])  # → (B, horizon * n_targets)
        return out.view(-1, self.horizon, self.n_targets)

# ── Dataset ──────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, df, window, horizon, target_scaler=None, feature_scaler=None):
        missing = [c for c in SEQ_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"CRITICAL: Dataframe missing required TCN features: {missing}")

        vals = df[SEQ_FEATURES].values.astype(np.float32)
        tgt  = df[TARGETS].values.astype(np.float32)

        # Apply feature scaler if provided (standardize all SEQ_FEATURES)
        if feature_scaler is not None:
            vals = feature_scaler.transform(vals).astype(np.float32)

        # Apply target scaler if provided (scale targets for balanced MSE)
        if target_scaler is not None:
            tgt = target_scaler.transform(tgt).astype(np.float32)

        self.X, self.y = [], []
        for i in range(len(vals) - window - horizon + 1):
            self.X.append(vals[i:i+window])
            self.y.append(tgt[i+window:i+window+horizon])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

        # Keep inverse transform for metrics reporting
        self.target_scaler = target_scaler

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])

    def inverse_transform_targets(self, preds):
        """Convert scaled predictions back to original MW units."""
        if self.target_scaler is not None:
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy()
            return self.target_scaler.inverse_transform(
                preds.reshape(-1, len(TARGETS))
            ).reshape(preds.shape)
        return preds

# ── Train ─────────────────────────────────────────────────────────────────────

def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    tc = cfg["model"]["tcn"]

    window, horizon = tc["window_size"], tc["forecast_horizon"]

    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df   = pd.read_parquet("data/processed/val.parquet")

    # ── Fix 1: Fit a target scaler so MSE is balanced across targets ──
    target_scaler = joblib.load("data/processed/target_scaler.pkl") if os.path.exists("data/processed/target_scaler.pkl") else None
    if target_scaler is None:
        from sklearn.preprocessing import StandardScaler
        target_scaler = StandardScaler().fit(train_df[TARGETS].values)
        os.makedirs("data/processed", exist_ok=True)
        joblib.dump(target_scaler, "data/processed/target_scaler.pkl")
        print("Saved target_scaler.pkl")

    # ── Fix 3: Fit a feature scaler on all SEQ_FEATURES (not just weather) ──
    feature_scaler = joblib.load("data/processed/tcn_scaler.pkl") if os.path.exists("data/processed/tcn_scaler.pkl") else None
    if feature_scaler is None:
        from sklearn.preprocessing import StandardScaler
        feature_scaler = StandardScaler().fit(train_df[SEQ_FEATURES].values)
        joblib.dump(feature_scaler, "data/processed/tcn_scaler.pkl")
        print("Saved tcn_scaler.pkl")

    train_ds = WindowDataset(train_df, window, horizon, target_scaler=target_scaler, feature_scaler=feature_scaler)
    val_ds   = WindowDataset(val_df,   window, horizon, target_scaler=target_scaler, feature_scaler=feature_scaler)
    train_dl = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=tc["batch_size"])

    device = get_device()
    print(f"Using device: {device}")
    model = TCN(len(SEQ_FEATURES), tc["filters"], tc["kernel_size"],
                tc["layers"], horizon, n_targets=len(TARGETS), dropout=tc.get("dropout", 0.2)).to(device)

    opt   = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])
    loss_fn = nn.MSELoss()

    # Early stopping
    patience = tc.get("early_stopping_patience", 15)
    patience_counter = 0

    best_val, best_state, best_epoch = float("inf"), None, 0
    with mlflow.start_run(run_name="tcn_multi_target_scaled"):
        mlflow.log_params({
            "targets": TARGETS,
            "window_size": window,
            "forecast_horizon": horizon,
            "filters": tc["filters"],
            "layers": tc["layers"],
            "n_features": len(SEQ_FEATURES),
            "target_scaling": "StandardScaler on train targets",
            "feature_scaling": "StandardScaler on all SEQ_FEATURES",
        })

        for epoch in range(1, tc["epochs"] + 1):
            model.train()
            train_loss = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                # Grad clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    val_loss += loss_fn(model(xb.to(device)), yb.to(device)).item()

            avg_val_loss = val_loss / len(val_dl)
            avg_train_loss = train_loss / len(train_dl)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            if avg_val_loss < best_val:
                best_val = avg_val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or patience_counter == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Best: {best_val:.6f} (ep {best_epoch})")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best val loss: {best_val:.6f} at epoch {best_epoch}")
                break

        # Log and Register the Best Model
        model.load_state_dict(best_state)
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="tcn_model",
            registered_model_name="GridTokenX_TCN"
        )

    model.load_state_dict(best_state)
    os.makedirs("models", exist_ok=True)
    torch.save({"state_dict": best_state,
                "config": tc,
                "in_features": len(SEQ_FEATURES),
                "n_targets": len(TARGETS),
                "targets": TARGETS}, "models/tcn.pt")
    print(f"Saved → models/tcn.pt (best epoch {best_epoch}, val_loss {best_val:.6f})")

if __name__ == "__main__":
    main()
