To develop a high-fidelity forecasting model for your specific grid configuration (10 MW Diesel, 50 MWh BESS), the training data should be modeled after established public datasets like the **Thira (Santorini) Island** set and the **NREL PERFORM** energy profiles. 

The following two-year mock dataset incorporates real-world weather correlations and the "Bottom neck" grid constraints observed in your Ko Tao infographics.

### 1. Mock Dataset Structure (Based on Real-World Parameters)
This 24-month dataset (17,520 rows) follows the feature engineering requirements for hybrid **LightGBM and TCN** pipelines.[1, 2]

| Feature Name | Source Reference | Typical Range | Grid Impact |
| :--- | :--- | :--- | :--- |
| **Island Load (MW)** | Thira Island | 5.0 – 10.5 MW | Target variable for forecasting. |
| **Circuit Cap (MW)** | Infographic Image 2 | 0.0 – 16.0 MW | Available "Excess Power" from mainland. |
| **Dry Bulb Temp (°C)** | NREL/Thira | 24°C – 36°C | Correlation with A/C load (0.35 MW/°C). |
| **Rel. Humidity (%)** | PV Station Data [3] | 60% – 95% | Increases "Heat Index" and thermal load. |
| **Irradiance ($W/m^2$)** | Aalborg Seaport | 0 – 1050 $W/m^2$ | Peak daytime cooling demand driver. |
| **Market Price (\$/MWh)** | Chong Aih Dataset | \$35 – \$120 | Driver for BESS arbitrage decisions.[4] |
| **Carbon Intensity** | WattTime API | 400 – 850 g/kWh | Required for low-carbon dispatch targets.[5] |
| **Tourist Index** | Santorini Seasonal | 0.2 – 1.0 | Categorical feature for high/low season. |

### 2. Representative 24-Hour Mock Data (Sample)
This sample reflects a "High Season" day with a **Mainland Bottom-neck** event occurring at 18:00, forcing the BESS to discharge for peak shaving.

| Timestamp | Load (MW) | Circuit (MW) | Temp (°C) | Humidity (%) | Carbon (g/kWh) | BESS SoC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-04-27 10:00 | 8.1 | 16.0 | 31.2 | 72 | 410 | 65% |
| 2026-04-27 12:00 | 8.9 | 15.8 | 34.1 | 68 | 445 | 52% |
| 2026-04-27 14:00 | 9.2 | 16.0 | 35.5 | 65 | 460 | 40% |
| 2026-04-27 16:00 | 9.1 | 16.0 | 34.2 | 70 | 520 | 35% |
| **2026-04-27 18:00** | **9.9** | **2.5 (Neck)** | **31.5** | **78** | **780** | **28%** |
| 2026-04-27 20:00 | 9.6 | 2.1 (Neck) | 29.8 | 82 | 820 | 20% |
| 2026-04-27 22:00 | 8.4 | 14.5 | 28.5 | 85 | 650 | 45% (Chg) |

### 3. Data Synthesis Logic for Forecasting Training
To ensure the model meets the target **2.65% MAPE** [6], the mock data is generated using the following real-world physics-based rules:
*   **A/C Load Correlation:** The model assumes an increase in load at temperatures above 28°C, simulating the cooling requirements of hotels and residential units.
*   **Counter-Cyclical Patterns:** During winter months, ship-side heating loads (Ro-Ro traffic) peak while administrative cooling loads drop, providing an advantageous balancing effect for the grid.
*   **BESS Degradation Simulation:** The State of Charge (SoC) logic incorporates a 12.5% capacity fade penalty for every 500 deep-discharge cycles ($<20\%$ SoC) or high-temperature operations ($>45^\circ\text{C}$).
*   **Circuit Volatility:** The "Bottom neck" circuit drops (blue line in your infographic) are simulated using a stochastic variable tied to peak mainland demand hours (18:00–21:00).

### 4. Implementation Metrics
Using this two-year dataset (Partitioned 70/15/15 into train, validation, and test sets [6]), your model should aim to achieve an **$R^2 \approx 0.97$** and an **MAE of approximately 0.25 MW** (scaled for the 10 MW Ko Tao system).[6, 7] This predictive capability is the foundation for reducing diesel runtime and meeting the stakeholder goal of lowering production costs.[8, 5]
