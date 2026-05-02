# GridTokenX Forecast ML — Implementation Plan
**Project:** Ko Tao Island Microgrid Predictive Dispatch
**Stack:** LightGBM + TCN hybrid | Python 3.11

---

## Directory Structure
```
gridtokenx-forecase-ml/
├── data/
│   ├── generate_dataset.py
│   └── preprocess.py
├── models/
│   ├── lgbm_model.py
│   ├── tcn_model.py
│   └── hybrid_pipeline.py
├── optimizer/
│   ├── dispatch.py
│   └── isca.py
├── evaluate.py
├── notebooks/
│   └── analysis.ipynb
├── config.yaml
└── requirements.txt
```

---

## Phase 1 — Data Layer

**Files:** `data/generate_dataset.py`, `data/preprocess.py`

### Auto Prompt
```
Generate a Python script `data/generate_dataset.py` that creates a synthetic 2-year hourly dataset (Jan 2025 – Dec 2026, 17520 rows) for Ko Tao island microgrid with these exact columns:

- Timestamp (hourly index)
- Island_Load_MW (target, 5.0–10.0 MW)
- Circuit_Cap_MW (0–16.0 MW, drops <5 MW stochastically at 18:00–22:00)
- Dry_Bulb_Temp (24–36°C)
- Rel_Humidity (60–95%)
- Solar_Irradiance (0–1050 W/m²)
- Wind_Speed (m/s)
- Cloud_Cover (%)
- Carbon_Intensity (400–850 g/kWh)
- Market_Price (35–120 USD/MWh)
- Tourist_Index (0.2–1.0)
- BESS_SoC_Pct (20–95%)
- Net_Delta_MW (derived: Island_Load_MW - Circuit_Cap_MW)

Physics rules:
- Load += 0.35 MW per °C above 28°C (A/C correlation)
- High season (Apr–Oct): base load +1.5 MW
- Circuit_Cap_MW bottleneck: stochastic drop <5 MW at 18:00–22:00 with 30% probability
- BESS charges (SoC increases) when Circuit_Cap > Load; discharges otherwise; clamp 20–95%

Save output to data/ko_tao_grid.parquet using pandas + pyarrow.
```

---

```
Generate a Python script `data/preprocess.py` that loads data/ko_tao_grid.parquet and:

1. Engineers features:
   - Lag features: Island_Load_MW at t-1h, t-24h, t-168h
   - Rolling mean and std of Island_Load_MW over 3h and 6h windows
   - Heat_Index = Dry_Bulb_Temp * Rel_Humidity / 100
   - Hour_of_Day, Day_of_Week, Is_Weekend, Is_High_Season (Apr–Oct=1)
   - Time_Until_High_Cap: hours until next Circuit_Cap_MW > 10 MW event

2. Splits chronologically (no shuffle):
   - Train: 70% (Jan 2025 – May 2026)
   - Val: 15% (Jun–Aug 2026)
   - Test: 15% (Sep–Dec 2026)

3. Normalizes numeric features using StandardScaler fitted on train only

4. Saves train/val/test to data/train.parquet, data/val.parquet, data/test.parquet
   and scaler to data/scaler.pkl
```

---

## Phase 2 — Models

**Files:** `models/lgbm_model.py`, `models/tcn_model.py`, `models/hybrid_pipeline.py`

### Auto Prompt
```
Generate `models/lgbm_model.py` that:
- Loads data/train.parquet and data/val.parquet
- Trains a LightGBM regressor on tabular/exogenous features:
  [Dry_Bulb_Temp, Rel_Humidity, Solar_Irradiance, Wind_Speed, Cloud_Cover,
   Carbon_Intensity, Market_Price, Tourist_Index, Circuit_Cap_MW,
   Hour_of_Day, Day_of_Week, Is_Weekend, Is_High_Season, Heat_Index]
- Target: Island_Load_MW
- Uses early stopping on val set (100 rounds)
- Saves model to models/lgbm.pkl
- Prints MAPE and R² on val set (target: MAPE < 2.65%, R² > 0.97)
```

---

```
Generate `models/tcn_model.py` using PyTorch that:
- Defines a TCN class with:
  - Causal 1D convolutions (no future leakage)
  - Dilated convolutions with dilation = 2^layer
  - 4 layers, 64 filters, kernel size 3
  - Residual connections
- Input: sliding window of 168 hours (1 week) of [Island_Load_MW lags, BESS_SoC_Pct, Net_Delta_MW]
- Output: next 24-hour Island_Load_MW forecast
- Trains on data/train.parquet, validates on data/val.parquet
- Saves model weights to models/tcn.pt
- Prints MAPE and R² on val set
```

---

```
Generate `models/hybrid_pipeline.py` that:
- Loads models/lgbm.pkl and models/tcn.pt
- For each sample: concatenates LightGBM prediction (scalar) with TCN 24h forecast
- Trains a lightweight linear meta-learner (Ridge regression) to combine both outputs
- Target: Island_Load_MW
- Saves meta-learner to models/meta_learner.pkl
- Prints final MAPE, MAE, R² on test set (data/test.parquet)
```

---

## Phase 3 — Optimization Engine

**Files:** `optimizer/dispatch.py`, `optimizer/isca.py`

### Auto Prompt
```
Generate `optimizer/dispatch.py` that implements predictive dispatch logic:

Given a 24-hour forecast array of Island_Load_MW and Circuit_Cap_MW:
1. Compute Net_Delta = Load - Circuit_Cap each hour
2. If Net_Delta <= 0: BESS charges, diesel off
3. If 0 < Net_Delta <= 7.9 MW: BESS discharges, diesel off
4. If Net_Delta > 7.9 MW: diesel fires at 7.5 MW (peak efficiency per BSFC curve),
   BESS covers remainder; excess diesel output recharges BESS
5. Enforce BESS SoC bounds 20–95%, capacity 50 MWh
6. Return hourly schedule: {diesel_MW, bess_mw, bess_soc, fuel_kg}

BSFC lookup: {10%: 350, 25%: 285, 50%: 225, 75%: 198.5, 90%: 205, 100%: 210} g/kWh
```

---

```
Generate `optimizer/isca.py` implementing the Improved Sine-Cosine Algorithm for dispatch optimization:

- Objective: minimize C_total = C_diesel + C_bess_degradation + C_carbon_penalty
  - C_diesel = fuel_kg * diesel_price_per_kg
  - C_bess_degradation = discharge_cycles * degradation_cost_per_cycle
  - C_carbon_penalty = CO2_kg * carbon_price_per_kg
- Decision variables: diesel on/off per hour, BESS dispatch MW per hour
- Constraints: load balance, BESS SoC 20–95%, diesel 0 or 7.5 MW
- Parameters: population=30, max_iter=200, elite retention=top 5
- Input: 24h forecast from hybrid_pipeline
- Output: optimal dispatch schedule dict
```

---

## Phase 4 — Evaluation

**File:** `evaluate.py`

### Auto Prompt
```
Generate `evaluate.py` that:
1. Loads data/test.parquet and runs the full hybrid pipeline (lgbm + tcn + meta-learner)
2. Runs dispatch.py with predicted vs actual load
3. Computes and prints:
   - MAPE, MAE, R² for load forecast
   - Fuel savings % vs reactive baseline (reactive = diesel always on at 50% load)
   - Carbon reduction %
   - BESS cycle count and estimated SoH after test period
4. Saves results to results/evaluation_report.json
```

---

## Phase 5 — Config & Requirements

### Auto Prompt
```
Generate `config.yaml` with all tunable parameters:
- data: start_date, end_date, bottleneck_probability, ac_coefficient (0.35), high_season_months, high_season_load_shift (1.5)
- bess: capacity_mwh (50), soc_min (0.20), soc_max (0.95), degradation_per_500_cycles (0.125)
- diesel: rated_mw (10), optimal_load_factor (0.75), bsfc_curve dict
- model: lgbm hyperparams, tcn layers/filters/kernel, window_size (168), forecast_horizon (24)
- optimizer: isca population (30), max_iter (200), diesel_price, carbon_price
- targets: mape (2.65), r2 (0.97), mae (0.25), fuel_savings (0.22)

Generate `requirements.txt` with pinned versions:
lightgbm==4.3.0, torch==2.3.0, pandas==2.2.2, numpy==1.26.4,
scikit-learn==1.4.2, pyarrow==16.0.0, fastparquet==2024.2.0,
pyyaml==6.0.1, matplotlib==3.8.4, notebook==7.1.3
```

---

## Execution Order

```bash
python data/generate_dataset.py      # Phase 1a
python data/preprocess.py            # Phase 1b
python models/lgbm_model.py          # Phase 2a
python models/tcn_model.py           # Phase 2b
python models/hybrid_pipeline.py     # Phase 2c
python evaluate.py                   # Phase 4
```

---

## Success Criteria

| Metric | Target |
|---|---|
| MAPE | < 2.65% |
| R² | > 0.97 |
| MAE | < 0.25 MW |
| Fuel savings | 20–25% |
| Carbon reduction | up to 55.1% |

---

## Phase 6 — Hyperparameter Optimization (Optuna)

**File:** `optimizer/tune.py`

### Auto Prompt
```
Generate `optimizer/tune.py` that uses Optuna to automate the search for optimal hyperparameters across the AI Predictive Control Layer.

1. **Objective Function:**
   - Ingest data/train.parquet and data/val.parquet.
   - Map "High Production Costs" pain point into a minimizing metric (MAPE).
   - Suggest hyperparameters for both LightGBM and TCN.

2. **Search Space:**
   - **LightGBM:** `num_leaves` (2–256), `learning_rate` (0.01–0.1), `lambda_l1`, `lambda_l2`.
   - **TCN:** `num_filters` (64–256), `kernel_size` (2–5), `dilation_base` (2).

3. **Multi-Objective (Pareto Front):**
   - Objective 1: Minimize MAPE (Precision for Diesel scheduling).
   - Objective 2: Minimize Training/Inference time (Responsiveness for Early Warning).

4. **Time-Series Validation:**
   - Use `TimeSeriesSplit` to prevent data leakage in the 17,520-row dataset.
   - Ensure the model is validated chronologically after the training set.

5. **Execution:**
   - Create a study `optuna.create_study(directions=["minimize", "minimize"])`.
   - Run 100 trials and save best parameters to config.yaml update.
```