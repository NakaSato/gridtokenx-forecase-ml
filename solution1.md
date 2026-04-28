Based on the grid architecture and historical load profiles provided for Ko Tao, here is a comprehensive mock dataset designed to train and validate the LightGBM and TCN forecasting pipelines. 

### 1. Ko Tao Island Grid Telemetry (24-Hour Sample)
This dataset represents the telemetry ingestion layer, combining the island's load (yellow line in your graphs) with circuit capacity (blue line) and exogenous environmental features.[1]

| Timestamp | Island Load (MW) | Circuit Capacity (MW) | Ambient Temp (°C) | Humidity (%) | BESS SoC (%) | Tourist Index (0-1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-03-27 00:00 | 6.45 | 14.2 | 28.2 | 82 | 85 | 0.85 |
| 2026-03-27 02:00 | 5.80 | 12.1 | 27.8 | 84 | 88 | 0.85 |
| 2026-03-27 04:00 | 5.42 | 11.5 | 27.5 | 85 | 92 | 0.85 |
| 2026-03-27 06:00 | 5.60 | 3.2 | 27.9 | 83 | 90 | 0.85 |
| 2026-03-27 08:00 | 7.15 | 16.8 | 29.5 | 78 | 75 | 0.85 |
| 2026-03-27 10:00 | 8.30 | 5.4 | 31.2 | 74 | 60 | 0.85 |
| 2026-03-27 12:00 | 8.75 | 18.2 | 33.5 | 70 | 45 | 0.85 |
| 2026-03-27 14:00 | 8.40 | 4.1 | 34.1 | 68 | 35 | 0.85 |
| 2026-03-27 16:00 | 8.55 | 15.5 | 32.8 | 72 | 40 | 0.85 |
| 2026-03-27 18:00 | 9.20 | 2.5 | 31.4 | 76 | 30 | 0.85 |
| 2026-03-27 20:00 | 9.85 | 12.8 | 30.1 | 79 | 20 | 0.85 |
| 2026-03-27 22:00 | 8.10 | 16.2 | 29.2 | 81 | 45 | 0.85 |

### 2. Diesel Generator Efficiency (BSFC) Lookup
For the Optimization Engine to calculate predictive dispatch, it requires the Brake Specific Fuel Consumption (BSFC) curve for the 10 MW unit. This data models the fuel waste associated with low-load "spinning reserve".

| Generator Load Factor (%) | Power Output (MW) | Specific Fuel Consumption (g/kWh) | Maintenance Impact Factor |
| :--- | :--- | :--- | :--- |
| 10% (Idle/Standby) | 1.0 | 350.0 | 2.5x (Carbonization risk) |
| 25% | 2.5 | 285.5 | 1.8x |
| 50% | 5.0 | 225.0 | 1.2x |
| 75% (Optimal) | 7.5 | 198.5 | 1.0x (Baseline) |
| 90% | 9.0 | 205.2 | 1.1x |
| 100% (Rated) | 10.0 | 210.0 | 1.3x |

### 3. BESS Health and Degradation Parameters
These parameters are used by the ML orchestrator to ensure the 50 MWh BESS is utilized in a way that maximizes State of Health (SoH).

*   **Optimal SoC Window:** 20% to 80% (extending cycle life by 30%).
*   **Capacity Fade Rate:** ~12.5% per 500 cycles at standard temperature (25°C).
*   **High-Temp Penalty:** Degradation increases to 20% fade per 500 cycles at 45°C.
*   **Internal Resistance:** Increases by ~8.7% over the first 500 cycles, impacting round-trip efficiency.

### 4. Target Forecasting Performance Metrics
When evaluating your LightGBM/TCN hybrid model against this data, use the following industrial benchmarks as your success criteria [1, 2]:

*   **MAPE (Mean Absolute Percentage Error):** Target < 2.65%.[1]
*   **MAE (Mean Absolute Error):** Target < 3.92 MW (for large grids) or scaled proportionally.[1]
*   **R² (Coefficient of Determination):** Target > 0.97.[2]
*   **Fuel Savings Potential:** Transitioning from reactive to predictive dispatch using this data typically targets a **20-25% reduction** in diesel consumption.[3]