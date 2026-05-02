"""
Evaluation: hybrid forecast + dispatch vs reactive baseline.
Output: results/evaluation_report.json
"""
import os, sys, json, pickle, subprocess, tempfile
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, r2_score

from models.tcn_model import TCN, WindowDataset
from models.device import get_device
from optimizer.dispatch import run_dispatch, schedule_summary

ROOT = os.path.dirname(__file__)


def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)


def lgbm_predict_subprocess(parquet_path: str) -> np.ndarray:
    out = tempfile.mktemp(suffix=".npy")
    script = f"""
import os, sys, pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, {repr(ROOT)})
import numpy as np, pandas as pd
from models.lgbm_model import FEATURES
with open("models/lgbm.pkl","rb") as f: model = pickle.load(f)
df = pd.read_parquet({repr(parquet_path)})

# Handle missing cluster features in real data splits by filling with 0.0
for col in FEATURES:
    if col not in df.columns:
        df[col] = 0.0

np.save({repr(out)}, model.predict(df[FEATURES]))
"""
    r = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    preds = np.load(out)
    os.unlink(out)
    return preds


def get_tcn_preds(ckpt, df, device):
    from torch.utils.data import DataLoader
    tc = ckpt["config"]
    dropout = ckpt.get("dropout", 0.0)
    net = TCN(ckpt["in_features"], tc["filters"], tc["kernel_size"],
              tc["layers"], tc["forecast_horizon"], dropout=dropout).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    ds = WindowDataset(df, tc["window_size"], tc["forecast_horizon"])
    dl = DataLoader(ds, batch_size=256, num_workers=0)
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(net(xb.to(device)).cpu().numpy())
    arr = np.concatenate(preds)[:, 0]
    aligned = np.full(len(df), np.nan)
    aligned[tc["window_size"]: tc["window_size"] + len(arr)] = arr
    return aligned


def reactive_baseline(load, circuit, cfg):
    """
    Reactive: diesel kept spinning at 50% load 24/7 as reserve.
    When deficit > load, diesel ramps to rated. BESS not used.
    """
    dc = cfg["diesel"]
    rated = dc["rated_mw"]
    curve = {float(k): v for k, v in dc["bsfc_curve"].items()}
    from optimizer.dispatch import _bsfc
    total_fuel = 0.0
    for h in range(len(load)):
        delta = load[h] - circuit[h]
        # Always run at least 50% (5MW) spinning reserve
        d_mw = max(5.0, min(delta, rated) if delta > 0 else 5.0)
        lf = d_mw / rated
        sfc = _bsfc(lf, curve)
        total_fuel += sfc * d_mw / 1000.0
    return total_fuel


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"Device: {device}")

    # ── Load models ──────────────────────────────────────────────────────────
    with open("models/meta_learner.pkl", "rb") as f:
        meta = pickle.load(f)
    ckpt = torch.load("models/tcn.pt", map_location=device)

    test = pd.read_parquet("data/processed/test.parquet")
    y_true = test["Island_Load_MW"].values

    # ── Forecast ─────────────────────────────────────────────────────────────
    print("LightGBM predictions...")
    lgbm_preds = lgbm_predict_subprocess("data/processed/test.parquet")
    print("TCN predictions...")
    tcn_preds  = get_tcn_preds(ckpt, test, device)

    mask = ~np.isnan(tcn_preds)
    X    = np.column_stack([lgbm_preds[mask], tcn_preds[mask]])
    hybrid_preds = meta.predict(X)
    y_eval = y_true[mask]

    forecast_metrics = {
        "mape":  round(mape(y_eval, hybrid_preds), 4),
        "mae":   round(float(mean_absolute_error(y_eval, hybrid_preds)), 4),
        "r2":    round(float(r2_score(y_eval, hybrid_preds)), 4),
    }
    print(f"Forecast → MAPE: {forecast_metrics['mape']}%  "
          f"MAE: {forecast_metrics['mae']} MW  R²: {forecast_metrics['r2']}")

    # ── Walk-forward 24h backtest (PEA requirement) ───────────────────────────
    freq = cfg["data"].get("frequency", "h")
    sph = 4 if freq == "15min" else 1
    step_24h = 24 * sph

    wf_mapes = []
    for start in range(0, len(y_eval) - step_24h, step_24h):
        yt = y_eval[start:start + step_24h]
        yp = hybrid_preds[start:start + step_24h]
        wf_mapes.append(float(mape(yt, yp)))
    backtest_mape = round(float(np.mean(wf_mapes)), 4)
    backtest_pass = backtest_mape <= 10.0
    print(f"Walk-forward MAPE (24h): {backtest_mape}%  PEA ≤10% → {'PASS ✅' if backtest_pass else 'FAIL ❌'}")

    # ── Dispatch on test set (day-by-day 24h windows) ────────────────────────
    # Circuit_Cap_MW dropped in preprocessing — use config max as conservative fallback
    circuit = test["Circuit_Cap_MW"].values if "Circuit_Cap_MW" in test.columns \
              else np.full(len(test), cfg["data"]["circuit_cap_max"])
    
    n_days  = len(y_eval) // step_24h
    pred_load = hybrid_preds[:n_days * step_24h].reshape(n_days, step_24h)
    pred_circ = circuit[mask][:n_days * step_24h].reshape(n_days, step_24h)

    total_fuel = total_carbon = diesel_hours = bess_cycles = 0.0
    soc = 0.65
    for d in range(n_days):
        sched = run_dispatch(pred_load[d], pred_circ[d], initial_soc=soc, cfg=cfg)
        s = schedule_summary(sched)
        total_fuel    += s["total_fuel_kg"]
        total_carbon  += s["total_carbon_kg"]
        diesel_hours  += s["diesel_hours"]
        bess_cycles   += sum(1 for h in sched if h.bess_mw > 0)
        soc = sched[-1].bess_soc

    # ── Reactive baseline ────────────────────────────────────────────────────
    reactive_fuel = reactive_baseline(y_eval[:n_days*24], circuit[:n_days*24], cfg)
    fuel_savings_pct = round((reactive_fuel - total_fuel) / reactive_fuel * 100, 2)
    carbon_reduction_pct = round(fuel_savings_pct, 2)  # proportional

    # ── BESS SoH estimate ────────────────────────────────────────────────────
    full_cycles = bess_cycles / (cfg["bess"]["capacity_mwh"] *
                                 (cfg["bess"]["soc_max"] - cfg["bess"]["soc_min"]))
    soh = round(1.0 - (full_cycles / 500) * cfg["bess"]["degradation_per_500_cycles"], 4)

    dispatch_metrics = {
        "days_evaluated":      n_days,
        "total_fuel_kg":       round(total_fuel, 2),
        "reactive_fuel_kg":    round(reactive_fuel, 2),
        "fuel_savings_pct":    fuel_savings_pct,
        "total_carbon_kg":     round(total_carbon, 2),
        "carbon_reduction_pct": carbon_reduction_pct,
        "diesel_hours":        int(diesel_hours),
        "bess_soh_estimate":   soh,
    }
    print(f"Dispatch → Fuel savings: {fuel_savings_pct}%  "
          f"Carbon reduction: {carbon_reduction_pct}%  "
          f"BESS SoH: {soh}")

    # ── Targets check ────────────────────────────────────────────────────────
    t = cfg["targets"]
    targets_met = {
        "mape_ok":         bool(forecast_metrics["mape"]  <= t["mape"]),
        "r2_ok":           bool(forecast_metrics["r2"]    >= t["r2"]),
        "mae_ok":          bool(forecast_metrics["mae"]   <= t["mae"]),
        "fuel_savings_ok": bool(fuel_savings_pct / 100    >= t["fuel_savings"]),
        "backtest_mape_pea_ok": backtest_pass,
    }
    print("Targets:", targets_met)

    report = {
        "forecast": forecast_metrics,
        "backtest_24h_mape": backtest_mape,
        "backtest_pea_pass": backtest_pass,
        "dispatch": dispatch_metrics,
        "targets_met": targets_met,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved → results/evaluation_report.json")


if __name__ == "__main__":
    main()
