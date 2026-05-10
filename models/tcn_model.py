"""
Multi-Target Temporal Convolutional Network for sequential forecasting.
Targets: tao_load_mw, cable_flow_mw, kmb_flow_mw.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import mlflow
from models.device import get_device
from models.mlflow_utils import setup_mlflow

setup_mlflow()
mlflow.set_experiment("GridTokenX_TCN")

# ── Feature Set (Aligned with Project Schema) ─────────────────────────────────

SEQ_FEATURES = [
    # 1. Per-location load + weather (8)
    "phangan_load_mw", "samui_load_mw", "phangan_t2m", "samui_t2m", 
    "t2m_celsius", "rh_pct", "ghi_w_m2", "wind_ms",
    
    # 2. System state (7)
    "headroom_mw", "max_capacity_mw", "capacity_mw", 
    "bess_soc_pct", "phangan_soc_pct", "samui_soc_pct", "tourist_index",
    
    # 3. Calendar (5) + Market (2)
    "hour_of_day", "day_of_week", "is_holiday", "is_songkran", "is_high_season",
    "carbon_intensity", "market_price",
    
    # 4. Critical Lags
    "tao_load_mw_lag_1h", "cable_flow_mw_lag_1h", "kmb_flow_mw_lag_1h",
    "kmb_trend", "kmb_seasonal", "kmb_resid"
]

TARGETS = ["tao_load_mw", "cable_flow_mw", "kmb_flow_mw"]

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
    def __init__(self, df, window, horizon):
        # Fill missing features with 0.0
        for col in SEQ_FEATURES:
            if col not in df.columns: df[col] = 0.0
        vals = df[SEQ_FEATURES].values.astype(np.float32)
        tgt  = df[TARGETS].values.astype(np.float32)
        self.X, self.y = [], []
        for i in range(len(vals) - window - horizon + 1):
            self.X.append(vals[i:i+window])
            self.y.append(tgt[i+window:i+window+horizon])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])

# ── Train ─────────────────────────────────────────────────────────────────────

def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    tc = cfg["model"]["tcn"]
    
    window, horizon = tc["window_size"], tc["forecast_horizon"]

    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df   = pd.read_parquet("data/processed/val.parquet")

    train_ds = WindowDataset(train_df, window, horizon)
    val_ds   = WindowDataset(val_df,   window, horizon)
    train_dl = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=tc["batch_size"])

    device = get_device()
    print(f"Using device: {device}")
    model = TCN(len(SEQ_FEATURES), tc["filters"], tc["kernel_size"],
                tc["layers"], horizon, n_targets=len(TARGETS), dropout=tc.get("dropout", 0.2)).to(device)
    
    opt   = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    with mlflow.start_run(run_name="tcn_multi_target_aligned"):
        mlflow.log_params({
            "targets": TARGETS,
            "window_size": window,
            "forecast_horizon": horizon,
            "filters": tc["filters"],
            "layers": tc["layers"],
        })

        for epoch in range(1, tc["epochs"] + 1):
            model.train()
            train_loss = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    val_loss += loss_fn(model(xb.to(device)), yb.to(device)).item()
            
            avg_val_loss = val_loss / len(val_dl)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("train_loss", train_loss / len(train_dl), step=epoch)

            if avg_val_loss < best_val:
                best_val, best_state = avg_val_loss, {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss/len(train_dl):.6f} | Val Loss: {avg_val_loss:.6f}")

    model.load_state_dict(best_state)
    os.makedirs("models", exist_ok=True)
    torch.save({"state_dict": best_state,
                "config": tc,
                "in_features": len(SEQ_FEATURES),
                "n_targets": len(TARGETS),
                "targets": TARGETS}, "models/tcn.pt")
    print("Saved → models/tcn.pt")

if __name__ == "__main__":
    main()
