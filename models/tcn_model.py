"""
Temporal Convolutional Network for sequential load forecasting.
Input:  data/train.parquet, data/val.parquet
Output: models/tcn.pt
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
from models.device import get_device

SEQ_FEATURES = ["Island_Load_MW", "BESS_SoC_Pct", "Net_Delta_MW",
                 "Load_Lag_1h", "Load_Lag_24h"]

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

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
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x) if self.downsample else x
        return torch.relu(out + res)


class TCN(nn.Module):
    def __init__(self, in_features, filters, kernel_size, n_layers, horizon):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = in_features if i == 0 else filters
            layers.append(TCNBlock(in_ch, filters, kernel_size, dilation=2**i))
        self.tcn = nn.Sequential(*layers)
        self.fc  = nn.Linear(filters, horizon)

    def forward(self, x):          # x: (B, T, F)
        x = x.permute(0, 2, 1)    # → (B, F, T)
        x = self.tcn(x)            # → (B, filters, T)
        return self.fc(x[:, :, -1])  # → (B, horizon)

# ── Dataset ──────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, df, window, horizon):
        vals = df[SEQ_FEATURES].values.astype(np.float32)
        tgt  = df["Island_Load_MW"].values.astype(np.float32)
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

    train_df = pd.read_parquet("data/train.parquet")
    val_df   = pd.read_parquet("data/val.parquet")

    train_ds = WindowDataset(train_df, window, horizon)
    val_ds   = WindowDataset(val_df,   window, horizon)
    train_dl = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=tc["batch_size"])

    device = get_device()
    print(f"Using device: {device}")
    model = TCN(len(SEQ_FEATURES), tc["filters"], tc["kernel_size"],
                tc["layers"], horizon).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                preds.append(model(xb.to(device)).cpu().numpy())
                trues.append(yb.numpy())
        preds = np.concatenate(preds).flatten()
        trues = np.concatenate(trues).flatten()
        val_mape = mape(trues, preds)

        if val_mape < best_val:
            best_val, best_state = val_mape, {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Val MAPE: {val_mape:.4f}%")

    model.load_state_dict(best_state)
    print(f"\nBest Val MAPE: {best_val:.4f}%  (target <2.65%)")

    os.makedirs("models", exist_ok=True)
    torch.save({"state_dict": best_state,
                "config": tc,
                "in_features": len(SEQ_FEATURES)}, "models/tcn.pt")
    print("Saved → models/tcn.pt")

if __name__ == "__main__":
    main()
