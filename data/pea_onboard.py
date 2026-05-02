"""
PEA Data Onboarding Pipeline
=============================
Handles distribution shift between synthetic training data and real PEA SCADA.

Steps:
  1. Schema mapping  — map PEA column names to GridTokenX schema
  2. Distribution check — compare PEA vs synthetic feature distributions
  3. Feature engineering — same pipeline as preprocess.py
  4. Scaler recalibration — refit StandardScaler on PEA calibration window
  5. Meta-learner refit — refit Ridge on PEA actuals (TCN+LGBM frozen)
  6. Backtest — walk-forward 24h MAPE on remaining PEA data

Usage:
    python data/pea_onboard.py --input data/raw/pea_telemetry_raw.csv
    python data/pea_onboard.py --input data/raw/pea_telemetry_raw.csv --calib-months 3
"""
import argparse, os, sys, pickle, subprocess, tempfile
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from data.preprocess import engineer_features, impute_bess_soc

ROOT = os.path.dirname(os.path.dirname(__file__))

# ── PEA column name → GridTokenX schema ──────────────────────────────────────
PEA_COLUMN_MAP = {
    # Common PEA SCADA register names (customise when actual CSV arrives)
    "PEA_LOAD_MW":        "Island_Load_MW",
    "CABLE_CAP_MW":       "Circuit_Cap_MW",
    "BESS_SOC":           "BESS_SoC_Pct",
    "TEMP_DRY":           "Dry_Bulb_Temp",
    "HUMIDITY":           "Rel_Humidity",
    "SOLAR_W_M2":         "Solar_Irradiance",
    "CARBON_INT":         "Carbon_Intensity",
    "MARKET_PRICE_THB":   "Market_Price",
    "TOURIST_IDX":        "Tourist_Index",
    # Passthrough if already correct names
    "Island_Load_MW":     "Island_Load_MW",
    "Circuit_Cap_MW":     "Circuit_Cap_MW",
    "BESS_SoC_Pct":       "BESS_SoC_Pct",
    "Dry_Bulb_Temp":      "Dry_Bulb_Temp",
    "Rel_Humidity":       "Rel_Humidity",
    "Solar_Irradiance":   "Solar_Irradiance",
    "Carbon_Intensity":   "Carbon_Intensity",
    "Market_Price":       "Market_Price",
    "Tourist_Index":      "Tourist_Index",
}

REQUIRED_COLS = ["Island_Load_MW", "Dry_Bulb_Temp", "Rel_Humidity"]


def load_cfg():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


# ── Step 1: Schema mapping ────────────────────────────────────────────────────

def map_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in PEA_COLUMN_MAP.items() if k in df.columns})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"PEA data missing required columns after mapping: {missing}\n"
                         f"Available: {list(df.columns)}\n"
                         f"Update PEA_COLUMN_MAP in data/pea_onboard.py")
    # Fill optional columns with synthetic defaults if absent
    if "Circuit_Cap_MW" not in df.columns:
        print("  ⚠️  Circuit_Cap_MW not found — using config max (conservative)")
        cfg = load_cfg()
        df["Circuit_Cap_MW"] = cfg["data"]["circuit_cap_max"]
    if "BESS_SoC_Pct" not in df.columns:
        df["BESS_SoC_Pct"] = 65.0
    for col, default in [("Solar_Irradiance", 200.0), ("Carbon_Intensity", 0.5),
                          ("Market_Price", 3.5), ("Tourist_Index", 0.7)]:
        if col not in df.columns:
            print(f"  ⚠️  {col} not found — using default {default}")
            df[col] = default
    return df


# ── Step 2: Distribution check ────────────────────────────────────────────────

def distribution_check(pea_df: pd.DataFrame, syn_path: str) -> dict:
    syn = pd.read_parquet(syn_path)
    check_cols = ["Island_Load_MW", "Dry_Bulb_Temp", "Rel_Humidity"]
    report = {}
    print("\n  Distribution Check (PEA vs Synthetic)")
    print(f"  {'Feature':<25} {'PEA mean':>10} {'Syn mean':>10} "
          f"{'PEA std':>9} {'Syn std':>9} {'Shift':>8}")
    print("  " + "─" * 75)
    for col in check_cols:
        if col not in pea_df.columns or col not in syn.columns:
            continue
        p_mean, p_std = pea_df[col].mean(), pea_df[col].std()
        s_mean, s_std = syn[col].mean(), syn[col].std()
        shift = abs(p_mean - s_mean) / (s_std + 1e-8)
        flag = "⚠️ " if shift > 1.0 else "✅"
        print(f"  {col:<25} {p_mean:>10.3f} {s_mean:>10.3f} "
              f"{p_std:>9.3f} {s_std:>9.3f} {shift:>7.2f}σ {flag}")
        report[col] = {"pea_mean": round(p_mean, 3), "syn_mean": round(s_mean, 3),
                       "shift_sigma": round(shift, 3)}
    return report


# ── Step 3+4: Feature engineering + scaler recalibration ─────────────────────

def recalibrate_scaler(calib_df: pd.DataFrame, cfg: dict) -> tuple:
    """Refit scaler on PEA calibration window. Returns (fe_df, new_scaler)."""
    df = impute_bess_soc(calib_df, cfg)
    df = engineer_features(df, cfg)

    exclude = {
        "Island_Load_MW", "Is_High_Season", "Hour_of_Day", "Day_of_Week",
        "Load_Lag_1h", "Load_Lag_24h", "Load_Lag_168h",
        "Load_Roll_Mean_3h", "Load_Roll_Std_3h",
        "Load_Roll_Mean_6h", "Load_Roll_Std_6h",
    }
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in exclude]

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled.loc[:, num_cols] = scaler.fit_transform(df[num_cols])

    return df_scaled, scaler, num_cols


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler,
                 num_cols: list, cfg: dict) -> pd.DataFrame:
    df = impute_bess_soc(df, cfg)
    df = engineer_features(df, cfg)
    df_scaled = df.copy()
    cols_present = [c for c in num_cols if c in df.columns]
    df_scaled.loc[:, cols_present] = scaler.transform(df[cols_present])
    return df_scaled


# ── Step 5: Meta-learner refit ────────────────────────────────────────────────

def lgbm_predict(df: pd.DataFrame) -> np.ndarray:
    tmp_path = tempfile.mktemp(suffix=".parquet")
    df.to_parquet(tmp_path)
    out = tempfile.mktemp(suffix=".npy")
    script = f"""
import os, sys, pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, {repr(ROOT)})
import numpy as np, pandas as pd
from models.lgbm_model import FEATURES
with open("models/lgbm.pkl","rb") as f: model = pickle.load(f)
df = pd.read_parquet({repr(tmp_path)})
avail = [c for c in FEATURES if c in df.columns]
X = df[avail].values
np.save({repr(out)}, model.predict(X))
"""
    r = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, cwd=ROOT)
    os.unlink(tmp_path)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    preds = np.load(out); os.unlink(out)
    return preds


def tcn_predict(df: pd.DataFrame) -> np.ndarray:
    import torch
    from models.tcn_model import TCN, SEQ_FEATURES
    from models.device import get_device

    ckpt   = torch.load(os.path.join(ROOT, "models/tcn.pt"),
                        map_location=get_device(), weights_only=False)
    tc     = ckpt["config"]
    device = get_device()
    net    = TCN(ckpt["in_features"], tc["filters"], tc["kernel_size"],
                 tc["layers"], tc["forecast_horizon"],
                 dropout=ckpt.get("dropout", 0.0)).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    seq_fields = [f.lower() for f in SEQ_FEATURES]
    avail = [f for f in seq_fields if f in df.columns]
    arr   = df[avail].values.astype(np.float32)
    W     = tc["window_size"]
    preds = np.full(len(df), np.nan)

    with torch.no_grad():
        for i in range(W, len(arr)):
            x = torch.tensor(arr[i-W:i]).unsqueeze(0).to(device)
            preds[i] = net(x).cpu().numpy()[0, 0]
    return preds


def refit_meta_learner(calib_scaled: pd.DataFrame) -> Ridge:
    """Refit Ridge meta-learner on PEA calibration data (TCN+LGBM frozen)."""
    print("  Running LGBM on calibration data...")
    lgbm_preds = lgbm_predict(calib_scaled)
    print("  Running TCN on calibration data...")
    tcn_preds  = tcn_predict(calib_scaled)

    mask = ~np.isnan(tcn_preds)
    X    = np.column_stack([lgbm_preds[mask], tcn_preds[mask]])
    y    = calib_scaled["Island_Load_MW"].values[mask]

    meta = Ridge(alpha=1.0)
    meta.fit(X, y)
    print(f"  Meta-learner refit: LGBM weight={meta.coef_[0]:.3f}, "
          f"TCN weight={meta.coef_[1]:.3f}, intercept={meta.intercept_:.3f}")
    return meta, mask


# ── Step 6: Backtest ──────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)


def backtest(backtest_scaled: pd.DataFrame, meta: Ridge) -> dict:
    print("  Running LGBM on backtest data...")
    lgbm_preds = lgbm_predict(backtest_scaled)
    print("  Running TCN on backtest data...")
    tcn_preds  = tcn_predict(backtest_scaled)

    mask = ~np.isnan(tcn_preds)
    X    = np.column_stack([lgbm_preds[mask], tcn_preds[mask]])
    hybrid = meta.predict(X)
    y_true = backtest_scaled["Island_Load_MW"].values[mask]

    overall_mape = mape(y_true, hybrid)
    overall_mae  = float(mean_absolute_error(y_true, hybrid))
    overall_r2   = float(r2_score(y_true, hybrid))

    # Walk-forward 24h windows
    wf_mapes = []
    for start in range(0, len(y_true) - 24, 24):
        wf_mapes.append(mape(y_true[start:start+24], hybrid[start:start+24]))

    # Error by hour-of-day (residual analysis)
    ts = backtest_scaled.index[mask]
    err_by_hour = {}
    for h in range(24):
        idx = [i for i, t in enumerate(ts) if t.hour == h]
        if idx:
            err_by_hour[h] = round(mape(y_true[idx], hybrid[idx]), 3)

    return {
        "n_samples":        int(mask.sum()),
        "mape_pct":         round(overall_mape, 4),
        "mae_mw":           round(overall_mae, 4),
        "r2":               round(overall_r2, 4),
        "backtest_24h_mape": round(float(np.mean(wf_mapes)), 4),
        "pea_target_pass":  overall_mape <= 10.0,
        "mape_by_hour":     err_by_hour,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PEA data onboarding pipeline")
    parser.add_argument("--input", default="data/raw/pea_telemetry_raw.csv",
                        help="Path to PEA CSV or Parquet file")
    parser.add_argument("--calib-months", type=int, default=3,
                        help="Months of data used for scaler+meta refit (rest = backtest)")
    parser.add_argument("--no-refit", action="store_true",
                        help="Skip meta-learner refit (use existing models/meta_learner.pkl)")
    args = parser.parse_args()

    cfg = load_cfg()

    # ── Load PEA data ─────────────────────────────────────────────────────────
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        print("   Place PEA SCADA export at that path and re-run.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  PEA ONBOARDING PIPELINE")
    print(f"  Input: {args.input}")
    print(f"  Calibration window: {args.calib_months} months")
    print(f"{'='*60}")

    ext = os.path.splitext(args.input)[1].lower()
    raw = pd.read_csv(args.input) if ext == ".csv" else pd.read_parquet(args.input)

    # Parse timestamp index
    ts_col = next((c for c in raw.columns if "time" in c.lower() or "date" in c.lower()), None)
    if ts_col:
        raw.index = pd.to_datetime(raw[ts_col])
        raw = raw.drop(columns=[ts_col])
    raw = raw.sort_index()
    raw.index = raw.index.tz_localize(None)  # strip tz if present

    print(f"\n  Loaded {len(raw):,} rows: {raw.index[0]} → {raw.index[-1]}")

    # ── Step 1: Schema mapping ────────────────────────────────────────────────
    print("\n[1/5] Schema mapping...")
    df = map_schema(raw)
    print(f"  ✅ Mapped to GridTokenX schema ({len(df.columns)} columns)")

    # ── Step 2: Distribution check ────────────────────────────────────────────
    print("\n[2/5] Distribution check...")
    syn_path = cfg["data"]["output_path"]
    dist_report = distribution_check(df, syn_path) if os.path.exists(syn_path) else {}

    # ── Split calibration / backtest ──────────────────────────────────────────
    calib_end = df.index[0] + pd.DateOffset(months=args.calib_months)
    calib_df  = df[df.index < calib_end]
    back_df   = df[df.index >= calib_end]

    print(f"\n  Calibration: {len(calib_df):,} rows ({calib_df.index[0].date()} → {calib_df.index[-1].date()})")
    print(f"  Backtest:    {len(back_df):,} rows ({back_df.index[0].date()} → {back_df.index[-1].date()})")

    if len(calib_df) < 200:
        print("  ⚠️  Calibration window < 200 rows — results may be unreliable")

    # ── Steps 3+4: Feature engineering + scaler recalibration ────────────────
    print("\n[3/5] Feature engineering + scaler recalibration...")
    calib_scaled, new_scaler, num_cols = recalibrate_scaler(calib_df, cfg)
    back_scaled = apply_scaler(back_df, new_scaler, num_cols, cfg)
    print(f"  ✅ Scaler refit on {len(calib_scaled):,} PEA rows")

    # ── Step 5: Meta-learner refit ────────────────────────────────────────────
    if args.no_refit:
        print("\n[4/5] Skipping meta-learner refit (--no-refit)")
        with open(os.path.join(ROOT, "models/meta_learner.pkl"), "rb") as f:
            meta = pickle.load(f)
    else:
        print("\n[4/5] Meta-learner refit on calibration data...")
        meta, _ = refit_meta_learner(calib_scaled)

    # ── Step 6: Backtest ──────────────────────────────────────────────────────
    print("\n[5/5] Walk-forward backtest on remaining PEA data...")
    results = backtest(back_scaled, meta)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS ({results['n_samples']:,} samples)")
    print(f"  MAPE:            {results['mape_pct']:.4f}%")
    print(f"  MAE:             {results['mae_mw']:.4f} MW")
    print(f"  R²:              {results['r2']:.4f}")
    print(f"  24h walk-forward MAPE: {results['backtest_24h_mape']:.4f}%")
    print(f"  PEA ≤10% target: {'PASS ✅' if results['pea_target_pass'] else 'FAIL ❌'}")

    # Worst hours
    worst = sorted(results["mape_by_hour"].items(), key=lambda x: -x[1])[:5]
    print(f"\n  Worst hours (residual analysis):")
    for h, m in worst:
        print(f"    h{h:02d}:00  MAPE={m:.2f}%")
    print(f"{'='*60}\n")

    # ── Save artifacts ────────────────────────────────────────────────────────
    if not args.no_refit:
        # Backup original, save new
        import shutil
        orig = os.path.join(ROOT, "models/meta_learner.pkl")
        shutil.copy(orig, orig.replace(".pkl", "_pretrain.pkl"))
        with open(orig, "wb") as f:
            pickle.dump(meta, f)
        print("  Saved → models/meta_learner.pkl  (backup: meta_learner_pretrain.pkl)")

    scaler_path = os.path.join(ROOT, "data/processed/scaler_pea.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(new_scaler, f)
    print(f"  Saved → data/processed/scaler_pea.pkl")

    import json
    report = {
        "input": args.input,
        "calib_months": args.calib_months,
        "distribution_shift": dist_report,
        "backtest": results,
    }
    out_path = os.path.join(ROOT, "results/pea_onboard_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved → results/pea_onboard_report.json\n")


if __name__ == "__main__":
    main()
