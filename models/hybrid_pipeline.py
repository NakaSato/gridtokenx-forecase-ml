"""
Hybrid pipeline: LightGBM + TCN → Ridge meta-learner.
LightGBM predictions are computed in an isolated subprocess to avoid
the OpenMP conflict between LightGBM and PyTorch on macOS.
"""
import os, sys, pickle, subprocess, tempfile
import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

mlflow.set_experiment("GridTokenX_Hybrid")

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def lgbm_predict_subprocess(parquet_path: str, out_path: str):
    """Run lgbm.predict in a clean subprocess, save result as .npy."""
    script = f"""
import os, sys, pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, {repr(ROOT)})
import numpy as np, pandas as pd
from models.lgbm_model import FEATURES
with open("models/lgbm.pkl", "rb") as f:
    model = pickle.load(f)
df = pd.read_parquet({repr(parquet_path)})

# Handle missing cluster features in real data splits by filling with 0.0
for col in FEATURES:
    if col not in df.columns:
        df[col] = 0.0

preds = model.predict(df[FEATURES])
np.save({repr(out_path)}, preds)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=ROOT
    )
    if result.returncode != 0:
        raise RuntimeError(f"lgbm subprocess failed:\n{result.stderr}")


def get_tcn_preds(ckpt, df, device):
    import torch
    from torch.utils.data import DataLoader
    from models.tcn_model import TCN, WindowDataset
    tc = ckpt["config"]
    dropout = ckpt.get("dropout", 0.0)
    window, horizon = tc["window_size"], tc["forecast_horizon"]
    net = TCN(ckpt["in_features"], tc["filters"], tc["kernel_size"],
              tc["layers"], horizon, dropout=dropout).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    ds = WindowDataset(df, window, horizon)
    dl = DataLoader(ds, batch_size=256, num_workers=0)
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(net(xb.to(device)).cpu().numpy())
    preds = np.concatenate(preds)[:, 0]
    aligned = np.full(len(df), np.nan)
    aligned[window: window + len(preds)] = preds
    return aligned


def main():
    import torch
    from models.device import get_device

    device = get_device()
    print(f"Using device: {device}")

    ckpt = torch.load("models/tcn.pt", map_location=device)

    val  = pd.read_parquet("data/processed/val.parquet")
    test = pd.read_parquet("data/processed/test.parquet")

    # LightGBM predictions via isolated subprocess
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        val_npy = f.name
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        test_npy = f.name

    print("Running LightGBM predictions (subprocess)...")
    lgbm_predict_subprocess("data/processed/val.parquet",  val_npy)
    lgbm_predict_subprocess("data/processed/test.parquet", test_npy)
    lgbm_val  = np.load(val_npy)
    lgbm_test = np.load(test_npy)
    os.unlink(val_npy); os.unlink(test_npy)
    print("LightGBM done.")

    # TCN predictions
    print("Running TCN predictions...")
    tcn_val  = get_tcn_preds(ckpt, val,  device)
    tcn_test = get_tcn_preds(ckpt, test, device)
    print("TCN done.")

    # Meta-learner
    def build(lgbm_p, tcn_p):
        mask = ~np.isnan(tcn_p)
        return np.column_stack([lgbm_p[mask], tcn_p[mask]]), mask

    X_val,  mv = build(lgbm_val,  tcn_val)
    X_test, mt = build(lgbm_test, tcn_test)
    y_val  = val["Island_Load_MW"].values[mv]
    y_test = test["Island_Load_MW"].values[mt]

    # Tune Ridge alpha
    best_a, best_m = 1.0, float("inf")
    for a in [0.01, 0.1, 1.0, 10.0, 100.0]:
        m = Ridge(alpha=a)
        m.fit(X_val, y_val)
        p = m.predict(X_val)
        val_m = mape(y_val, p)
        if val_m < best_m:
            best_m, best_a = val_m, a
    
    print(f"Best Ridge alpha: {best_a} (Val MAPE: {best_m:.4f}%)")
    meta = Ridge(alpha=best_a)
    meta.fit(X_val, y_val)
    preds = meta.predict(X_test)

    test_mape = mape(y_test, preds)
    test_mae  = mean_absolute_error(y_test, preds)
    test_r2   = r2_score(y_test, preds)

    print(f"\nTest MAPE : {test_mape:.4f}%  (target <10.0%)")
    print(f"Test MAE  : {test_mae:.4f} MW  (target <0.75)")
    print(f"Test R²   : {test_r2:.4f}  (target >0.30)")

    # ── Walk-forward 24h backtest (PEA requirement) ───────────────────────────
    print("\n── Walk-forward 24h Backtest ──")
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    freq = cfg["data"].get("frequency", "h")
    sph = 4 if freq == "15min" else 1
    step_24h = 24 * sph

    wf_mapes = []
    for start in range(0, len(y_test) - step_24h, step_24h):
        yt = y_test[start:start + step_24h]
        yp = preds[start:start + step_24h]
        if len(yt) == step_24h:
            wf_mapes.append(mape(yt, yp))
    backtest_mape = float(np.mean(wf_mapes))
    print(f"Walk-forward MAPE (24h windows): {backtest_mape:.4f}%  (PEA target ≤10%)")
    print(f"Windows evaluated: {len(wf_mapes)}")
    print(f"PASS: {backtest_mape <= 10.0}")

    with mlflow.start_run(run_name="hybrid_train"):
        mlflow.log_params({"meta_alpha": 1.0})
        mlflow.log_metrics({
            "test_mape": test_mape,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "backtest_mape_24h": backtest_mape,
        })
        from mlflow.models import infer_signature
        signature = infer_signature(X_test, preds)
        input_example = pd.DataFrame(X_test[:5], columns=["lgbm_pred", "tcn_pred"])
        if not os.environ.get("COLAB_TRAIN"):
            mlflow.sklearn.log_model(meta, "meta_learner", signature=signature, input_example=input_example)

    os.makedirs("models", exist_ok=True)
    with open("models/meta_learner.pkl", "wb") as f:
        pickle.dump(meta, f)
    print("Saved → models/meta_learner.pkl")


if __name__ == "__main__":
    main()
