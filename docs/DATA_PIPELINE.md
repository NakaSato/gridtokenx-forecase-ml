# GridTokenX: Data Pipeline Documentation

The GridTokenX data pipeline is a unified orchestrator for ingesting, integrating, and preprocessing power grid telemetry and environmental data. It supports three primary modes of operation: Synthetic Generation, Real-World ERA5 Integration, and PEA AWS Sandbox Integration.

## Architecture

The pipeline is managed by `data/pipeline.py` and is organized into three sequential stages:

1.  **Ingestion/Integration**: Gathering raw data from various sources (synthetic or real).
2.  **Preprocessing**: Feature engineering, scaling, and time-series decomposition.
3.  **Validation**: Automated data quality checks (null values, monotonicity, cluster feature availability).

## Usage

The recommended way to run the pipeline is via the `just` task runner.

### 1. Synthetic Generation (Default)
Generates high-fidelity research data for the Ko Tao-Phangan-Samui cluster using physics-aware simulation.
```bash
just data-pipeline --force
```

### 2. PEA AWS Sandbox Integration
Integrates raw SCADA exports from the PEA AWS environment.
```bash
just data-pipeline --pea /path/to/raw/scada/dir
```

### 3. Real-World Integration (ERA5 + Tourism)
Uses local raw parquets (`data/raw/`) containing historical weather and tourism indices.
```bash
just data-pipeline --real
```

### 4. Fetching Public Datasets
Downloads required environmental and renewable energy datasets before processing.
```bash
just data-pipeline --fetch
```

## Configuration

Pipeline behavior is controlled via `config.yaml` under the `data`, `cluster`, and `bess` blocks:

- **Time Range**: `data.start_date` to `data.end_date`.
- **Frequency**: Default `15min`.
- **Holidays**: Custom Thai holiday dates for Songkran stress tests.
- **Island Assets**: Ratings for BESS, Diesel, and load bases per island.

## Feature Schema

The pipeline produces a unified dataset with 40+ features, including:

| Category | Features |
| :--- | :--- |
| **Load** | `Island_Load_MW`, `Phangan_Load_MW`, `Samui_Load_MW` |
| **Grid State** | `Circuit_Cap_MW`, `Samui_Circuit_MW`, `BESS_SoC_Pct` |
| **Weather** | `Dry_Bulb_Temp`, `Rel_Humidity`, `Solar_Irradiance`, `Wind_Speed` |
| **Economic** | `Market_Price`, `Carbon_Intensity`, `Tourist_Index` |
| **Temporal** | `Is_Thai_Holiday`, `Is_Songkran`, `hour_sin`, `month_cos` |

## Output Artifacts

- **Consolidated Data**: `data/processed/island_grid.parquet`
- **Model Splits**: `data/processed/train.parquet`, `val.parquet`, `test.parquet`
- **Normalization**: `data/processed/scaler.pkl` (StandardScaler)

## Automated Validation

At the end of every run, the pipeline validates:
- **Integrity**: No missing splits or critical columns.
- **Quality**: Zero null values and monotonic timestamps.
- **Cluster Fidelity**: Checks that cluster-mode features (like Samui SoC) are not degenerate.
