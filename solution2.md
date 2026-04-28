Based on the specific infrastructure and operational limits provided in the infographic for the **Ko Tao** island grid, I have developed a comprehensive mock dataset. This data simulates a 24-hour cycle where the "Bottom neck" constraint and "Excess Power" availability force critical dispatch decisions.

### 1. Grid Component Reference (from Infographic)
*   **Total Island Load:** 5.0 MW to 10.0 MW range.
*   **Import Capacity (Excess Power):** 0 MW to 16.0 MW (variable based on Samui/Phangan demand).
*   **Generation Asset:** 10 MW Diesel Unit (Item 8).
*   **Storage Asset:** 50 MWh Battery Energy Storage System (Item 7).
*   **The Constraint:** The 115 kV "Bottom neck" affects the stability of the 16 MW supply from the mainland (Khanom).[1]

### 2. Time-Series Telemetry Dataset (24-Hour Cycle)
This dataset includes the features required for a **hybrid LightGBM (tabular/categorical)** and **TCN (sequential)** forecasting architecture.[2]

| Timestamp | Ko Tao Load (MW) | Mainland Supply Available (MW) | Net Load Delta (MW) | Diesel Gen Output (MW) | BESS Charge (-) / Discharge (+) (MW) | BESS SoC (%) | Ambient Temp (°C) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 00:00 | 6.2 | 16.0 | -9.8 | 0.0 | -8.0 (Charging) | 45.0 | 27.5 |
| 03:00 | 5.1 | 16.0 | -10.9 | 0.0 | -8.0 (Charging) | 88.0 | 26.8 |
| 06:00 | 5.8 | 14.5 | -8.7 | 0.0 | -2.0 (Topping) | 95.0 | 27.2 |
| 09:00 | 8.2 | 12.0 | -3.8 | 0.0 | 0.0 | 95.0 | 30.5 |
| 12:00 | 9.4 | 8.0 | 1.4 | 0.0 | 1.4 | 92.2 | 33.8 |
| 15:00 | 8.8 | 5.0 | 3.8 | 0.0 | 3.8 | 84.6 | 32.1 |
| 18:00 | 9.9 | 2.0 (Bottleneck) | 7.9 | 0.0 | 7.9 | 68.8 | 30.2 |
| 19:00 | 10.0 | 1.5 (Bottleneck) | 8.5 | 7.5 (Peak Eff) | -2.0 (Recharging) | 71.0 | 29.5 |
| 21:00 | 9.5 | 3.5 | 6.0 | 0.0 | 6.0 | 55.0 | 28.8 |
| 23:00 | 7.2 | 15.0 | -7.8 | 0.0 | -7.0 (Charging) | 69.0 | 28.1 |

### 3. ML Feature Engineering Metadata
To train the LightGBM model, the following categorical and engineered features should be derived from the raw telemetry [2, 3]:

*   **Load Lags (Sequential):** $t-1h, t-2h, t-24h$.
*   **Rolling Statistics:** 4-hour moving average of "Mainland Supply" to detect incoming bottleneck periods.
*   **Temporal Features:** `Hour_of_Day`, `Is_Weekend`, and `Tourist_Season_Peak` (Binary).
*   **Interaction Features:** `Temp_Humidity_Index` to predict air conditioning surges that drive the 10 MW peak load.[3]

### 4. Operational Logic for Optimization
The data above reflects a **Predictive Dispatch** strategy as discussed in the research:
1.  **BESS Priority:** The BESS (Item 7) covers the 1.4 MW to 7.9 MW deficit during the afternoon and early evening to avoid starting the diesel generator.[4]
2.  **Optimal Generator Loading:** At 19:00, when the bottleneck is severe (1.5 MW available) and the load is at max (10 MW), the system fires the 10 MW generator at its **7.5 MW peak efficiency point**. 
3.  **Arbitrage:** Excess power from the diesel generator at 19:00 (since $7.5\text{ Gen} + 1.5\text{ Grid} > 10.0\text{ Load}$) is used to **recharge the BESS**, effectively "storing" high-efficiency diesel energy for the next surge.[5, 6]

### 5. Performance Benchmarks for Model Validation
When running your TCN-LightGBM pipeline on this data, the following targets indicate an "expert-level" implementation [5, 3, 7]:
*   **Load Forecast Accuracy (MAPE):** < 2.65%.
*   **Net Load (Delta) Forecast Accuracy:** < 5.0% (due to higher volatility of import capacity).
*   **Fuel Savings Goal:** 20-25% reduction in diesel consumption compared to reactive (threshold-based) dispatch.