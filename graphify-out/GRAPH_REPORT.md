# Graph Report - gridtokenx-forecase-ml  (2026-05-10)

## Corpus Check
- 66 files · ~40,870 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 947 nodes · 1229 edges · 44 communities detected
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 197 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]

## God Nodes (most connected - your core abstractions)
1. `TCN` - 29 edges
2. `engineer_features()` - 21 edges
3. `run_dispatch()` - 21 edges
4. `IslandGrid` - 18 edges
5. `TestRunDispatch` - 16 edges
6. `useNetwork()` - 16 edges
7. `cn()` - 15 edges
8. `TestEngineerFeatures` - 13 edges
9. `SQLiteStreamingEngine` - 13 edges
10. `check_warnings()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `forecast()` --calls--> `schedule_summary()`  [INFERRED]
  infrastructure/api/routes/forecast.py → domain/dispatch.py
- `get_tcn_preds()` --calls--> `TCN`  [INFERRED]
  research/evaluate.py → models/tcn_model.py
- `main()` --calls--> `cluster_optimize()`  [INFERRED]
  research/evaluate.py → optimizer/pea_dispatch_opt.py
- `isca_optimize()` --calls--> `HourlyDispatch`  [INFERRED]
  optimizer/isca.py → domain/entities.py
- `objective()` --calls--> `TCN`  [INFERRED]
  optimizer/tune.py → models/tcn_model.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.04
Nodes (50): forecast(), _hybrid_forecast(), _lgbm_predict(), Run LightGBM prediction in a clean subprocess to avoid macOS OpenMP conflicts., Run LightGBM prediction in a clean subprocess to avoid macOS OpenMP conflicts., Run LightGBM prediction in a clean subprocess to avoid macOS OpenMP conflicts., stream_telemetry(), _tcn_predict() (+42 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (47): apply_scaler(), backtest(), distribution_check(), lgbm_predict(), load_cfg(), main(), map_schema(), mape() (+39 more)

### Community 2 - "Community 2"
Cohesion: 0.04
Nodes (16): MeterCard(), getMeterTheme(), useApi(), useLogs(), usePagination(), usePrices(), useWebSocket(), calculateEnergyMW() (+8 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (40): IslandGrid, Manages full state of a single island microgrid., ActualRequest, AgentActionPlanRequest, AgentExecutiveReportRequest, AgentExplainRequest, AgentForecastNarrativeRequest, AgentGridStatusRequest (+32 more)

### Community 4 - "Community 4"
Cohesion: 0.04
Nodes (30): client(), Tests for api/serve.py — FastAPI endpoint integration tests. Uses TestClient (no, Tests for POST /stream/actual., Tests for POST /stream/actual., Tests for GET /stream/metrics., Tests for GET /stream/metrics., Tests for POST /warnings., Tests for POST /warnings. (+22 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (33): warnings(), check_warnings(), format_warnings(), Early Warning System — detects impending BESS depletion / bottleneck events.  Lo, Run pandapower AC power flow for the given island loads.     Returns HVDC connec, Run pandapower AC power flow for the given island loads.     Returns HVDC connec, _run_physics_check(), Warning (+25 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (28): Dataset, get_tcn_preds(), lgbm_predict_subprocess(), main(), mape(), Evaluation: hybrid forecast + dispatch vs reactive baseline. Output: results/eva, Simple reactive dispatch (start diesel when load > circuit + margin)., reactive_baseline() (+20 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (20): ABC, get_config(), get_config_dep(), get_predictor(), get_predictor_dep(), get_streaming_engine(), get_streaming_engine_dep(), SQLiteStreamingEngine (+12 more)

### Community 8 - "Community 8"
Cohesion: 0.06
Nodes (20): GridDashboard(), MapLegend(), SearchFilterPanel(), useEgatTransmissionData(), useElectricalGridData(), useGridAssets(), useGridStats(), useKoTaoNetwork() (+12 more)

### Community 9 - "Community 9"
Cohesion: 0.06
Nodes (10): Regression tests for model predictions — ensures trained models produce consiste, Verify the system meets all PEA performance targets., Verify meta_learner.pkl was saved with the current sklearn version., Loading meta_learner.pkl should not produce InconsistentVersionWarning., Verify all required model files exist and are loadable., Verify all required data files exist., TestDataArtifactsExist, TestModelArtifactsExist (+2 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (24): IslandState, load_cfg(), Cluster Stress Test — 2026 Songkran Festival  Simulates the entire Ko Tao-Phanga, run_cluster_test(), load_cfg(), GridTokenX Intelligence Diagnostic — 2026 Strategy Audit Synthesizes Topology, P, run_diagnostic(), estimate_cluster_loads() (+16 more)

### Community 11 - "Community 11"
Cohesion: 0.08
Nodes (26): dispatch_tool(), forecast_tool(), Mock forecast tool returning standard load forecasts., Mock dispatch tool representing the MILP scheduler., Retrieves standard operating procedures relevant to the given query.     In a re, sop_tool(), generate_action_plan(), generate_decision_explanation() (+18 more)

### Community 12 - "Community 12"
Cohesion: 0.1
Nodes (18): dispatch_cluster(), Run ADMM multi-island diesel coordination., Run ADMM multi-island diesel coordination., Run ADMM multi-island diesel coordination., get_cluster_dispatch(), IslandAgent, Local step for Exchange Problem:         Minimize C_i * d_i + (rho/2) * || d_i -, ADMM for Cluster-wide Diesel Coordination (Exchange Problem).     Minimize sum(C (+10 more)

### Community 13 - "Community 13"
Cohesion: 0.13
Nodes (23): _bsfc_interp(), _build_cluster_milp(), _build_milp(), cluster_optimize(), ClusterStepResult, _fit_linear_fuel(), _fuel_cost(), HourResult (+15 more)

### Community 14 - "Community 14"
Cohesion: 0.12
Nodes (10): BESS, IslandGrid, MultiGenPlant, Core Library: Physical Grid Component Models. Encapsulates state and logic for B, Estimate I^2R losses in MW., Manages full state of a single island microgrid., Apply power request. Positive = Discharge, Negative = Charge.         Returns ac, Manages a plant of 5 identical 2MW diesel units. (+2 more)

### Community 15 - "Community 15"
Cohesion: 0.12
Nodes (9): BESS, BESSState, MultiGenPlant, Core Library: Physical Grid Component Models. Encapsulates state and logic for B, Estimate I^2R losses in MW., Apply power request. Positive = Discharge, Negative = Charge.         Returns ac, Manages a plant of 5 identical 2MW diesel units., Determines required units and delivers power. (+1 more)

### Community 16 - "Community 16"
Cohesion: 0.24
Nodes (14): _ar1(), _bess_soc(), generate(), generate_ko_phangan(), generate_ko_samui(), generate_ko_tao(), load_cfg(), main() (+6 more)

### Community 17 - "Community 17"
Cohesion: 0.19
Nodes (14): build_table(), detect_geom_type(), get_engine(), ingest_table(), load_and_rename(), main(), print_summary(), GridTokenX — Load GeoJSON spatial data into PostGIS ============================ (+6 more)

### Community 18 - "Community 18"
Cohesion: 0.14
Nodes (13): cfg(), high_deficit_load(), low_circuit(), Shared pytest fixtures for GridTokenX test suite., Load project config.yaml once per test session., Typical 24h load profile (MW) — realistic Ko Tao range., 24h circuit capacity — includes evening bottleneck window., 24h circuit capacity — no bottleneck (all surplus). (+5 more)

### Community 19 - "Community 19"
Cohesion: 0.16
Nodes (6): MapControls(), MapInfoCard(), MapLegend(), createCustomIcon(), getMeterColor(), getMeterSize()

### Community 20 - "Community 20"
Cohesion: 0.24
Nodes (11): get_tcn_preds(), lgbm_predict_subprocess(), main(), mape(), Hybrid pipeline: Multi-Target LightGBM + TCN → Parallel Ridge meta-learners. Ali, Run multi-target lgbm.predict in a clean subprocess., Run lgbm.predict in a clean subprocess, save result as .npy., Core meta-learner training logic. (+3 more)

### Community 21 - "Community 21"
Cohesion: 0.15
Nodes (2): proxyGET(), proxyPOST()

### Community 22 - "Community 22"
Cohesion: 0.15
Nodes (2): proxyGET(), proxyPOST()

### Community 23 - "Community 23"
Cohesion: 0.29
Nodes (10): circuit_for_hour(), circuit_forecast_for(), lgbm_features_for(), main(), print_header(), print_row(), Real-Time Simulator — streams test.parquet row-by-row to the API.  Every hour:, 24h circuit capacity forecast starting from pos (96 steps for 15min). (+2 more)

### Community 25 - "Community 25"
Cohesion: 0.33
Nodes (4): IslandNode, Alternating Direction Method of Multipliers (ADMM)     Ensures sum(p_export) = 0, Alternating Direction Method of Multipliers (ADMM)     Ensures sum(p_export) = 0, run_admm_consensus()

### Community 26 - "Community 26"
Cohesion: 0.33
Nodes (3): GemmaClient, Generates text from the LLM., Fallback mock generator for prototype demonstration without API keys.

### Community 28 - "Community 28"
Cohesion: 0.5
Nodes (4): load_scada_file(), main(), PEA AWS Sandbox Integration Script =================================== Aggregate, Load a single SCADA file (CSV or Parquet) and normalize index.

### Community 29 - "Community 29"
Cohesion: 0.4
Nodes (3): generate_kireip_proxy(), KIREIP Proxy Dataset — King Island Renewable Energy Integration Project Calibrat, Generate 2-year hourly proxy dataset.     scale: multiplier to map KI load (~2.5

### Community 30 - "Community 30"
Cohesion: 0.5
Nodes (4): map_pea_to_schema(), PEA Ground-Truth Validation Suite ================================== This script, Map proprietary PEA columns to GridTokenX schema.     Customize this function ba, run_commissioning_audit()

### Community 31 - "Community 31"
Cohesion: 0.67
Nodes (3): main(), GridTokenX: Master Training & Backtest Pipeline ================================, run_step()

### Community 32 - "Community 32"
Cohesion: 0.67
Nodes (3): main(), mape(), Multi-Target LightGBM model for tabular/exogenous feature forecasting. Targets:

### Community 33 - "Community 33"
Cohesion: 0.5
Nodes (1): Fetch public weather + solar irradiance data for Ko Tao, Phangan, Samui. Sources

### Community 34 - "Community 34"
Cohesion: 0.67
Nodes (3): main(), GridTokenX: Master Training & Backtest Pipeline ================================, run_step()

### Community 35 - "Community 35"
Cohesion: 0.67
Nodes (3): load_cfg(), Optimal Power Flow (OPF) Analysis — GridTokenX Determines the most cost-effectiv, run_opf_analysis()

### Community 36 - "Community 36"
Cohesion: 0.83
Nodes (3): fetchMetrics(), handleSubmit(), parseCSV()

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (2): generate_plots(), load_cfg()

### Community 39 - "Community 39"
Cohesion: 0.67
Nodes (1): GridTokenX — Data Folder Cleanup Utility =======================================

### Community 40 - "Community 40"
Cohesion: 0.67
Nodes (1): Download freely available microgrid datasets (no login required).

### Community 41 - "Community 41"
Cohesion: 0.67
Nodes (1): Download NREL ARPA-E PERFORM dataset (no login required). Fetches: Load actuals,

### Community 43 - "Community 43"
Cohesion: 0.67
Nodes (2): Analyzes how temperature shifts impact the total Island Load and      subsequent, run_sensitivity_study()

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): GridTokenX — Colab GPU Training Script Runs full pipeline: generate → preprocess

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Centralized schema definition for GridTokenX models. Ensures consistency between

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Recalibrate island_grid.parquet using real-world reference datasets:   - microgr

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): GridTokenX — Colab GPU Training Script Runs full pipeline: generate → preprocess

## Knowledge Gaps
- **226 isolated node(s):** `GridTokenX: Master Training & Backtest Pipeline ================================`, `Evaluation: multi-target hybrid forecast (3-island) + coordinated dispatch. Alig`, `GridTokenX — Colab GPU Training Script Runs full pipeline: generate → preprocess`, `GridTokenX Coordinated Cluster MILP Dispatch ===================================`, `Alternating Direction Method of Multipliers (ADMM)     Ensures sum(p_export) = 0` (+221 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 21`** (13 nodes): `GET()`, `GET()`, `route.ts`, `route.ts`, `route.ts`, `route.ts`, `proxy-utils.ts`, `route.ts`, `proxyGET()`, `proxyPOST()`, `POST()`, `GET()`, `GET()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (13 nodes): `GET()`, `POST()`, `route.ts`, `route.ts`, `route.ts`, `route.ts`, `proxy-utils.ts`, `route.ts`, `proxyGET()`, `proxyPOST()`, `GET()`, `GET()`, `POST()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (4 nodes): `fetch_nasa_power()`, `fetch_openmeteo()`, `fetch_public_datasets.py`, `Fetch public weather + solar irradiance data for Ko Tao, Phangan, Samui. Sources`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (3 nodes): `generate_plots()`, `load_cfg()`, `plot_2026_strategy.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (3 nodes): `cleanup()`, `cleanup.py`, `GridTokenX — Data Folder Cleanup Utility =======================================`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (3 nodes): `download()`, `fetch_public_microgrid.py`, `Download freely available microgrid datasets (no login required).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (3 nodes): `download()`, `fetch_nrel_perform.py`, `Download NREL ARPA-E PERFORM dataset (no login required). Fetches: Load actuals,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (3 nodes): `sensitivity_analysis.py`, `Analyzes how temperature shifts impact the total Island Load and      subsequent`, `run_sensitivity_study()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (2 nodes): `colab_train.py`, `GridTokenX — Colab GPU Training Script Runs full pipeline: generate → preprocess`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (2 nodes): `schema.py`, `Centralized schema definition for GridTokenX models. Ensures consistency between`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (2 nodes): `calibrate_with_real_data.py`, `Recalibrate island_grid.parquet using real-world reference datasets:   - microgr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (2 nodes): `colab_train.py`, `GridTokenX — Colab GPU Training Script Runs full pipeline: generate → preprocess`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TCN` connect `Community 3` to `Community 9`, `Community 20`, `Community 6`, `Community 1`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `tcn_predict()` connect `Community 1` to `Community 3`, `Community 6`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `warnings()` connect `Community 5` to `Community 3`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Are the 25 inferred relationships involving `TCN` (e.g. with `TestModelArtifactsExist` and `TestDataArtifactsExist`) actually correct?**
  _`TCN` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `engineer_features()` (e.g. with `.test_lag_features_created()` and `.test_rolling_features_created()`) actually correct?**
  _`engineer_features()` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `run_dispatch()` (e.g. with `main()` and `.test_output_length()`) actually correct?**
  _`run_dispatch()` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `IslandGrid` (e.g. with `TelemetryRow` and `TelemetryStreamRequest`) actually correct?**
  _`IslandGrid` has 13 INFERRED edges - model-reasoned connections that need verification._