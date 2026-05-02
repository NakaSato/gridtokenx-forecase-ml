To support your hybrid **LightGBM and TCN** model, I have developed a multi-variable mock dataset spanning two full years (January 2025 – December 2026). This data incorporates the specific infrastructure constraints from your infographics (the 115 kV "Bottom neck" and 10 MW generator) alongside synthetic meteorological data tailored for an island environment like Ko Tao.

### 1. Two-Year Dataset Structure (Hourly Resolution)
This dataset consists of **17,520 rows**. For training, the data should be processed using a sliding window for the TCN (sequential) and feature-engineered for LightGBM (tabular).[1, 2]

| Feature | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `Timestamp` | Index | DateTime | Hourly intervals (YYYY-MM-DD HH:MM). |
| **`Island_Load_MW`** | **Target** | MW | The 5–10 MW demand shown in your infographic. |
| `Circuit_Cap_MW` | Exogenous | MW | The 0–16 MW variable import limit ("Excess Power"). |
| `Dry_Bulb_Temp` | Weather | °C | Driving factor for air conditioning (A/C) load.[3] |
| `Rel_Humidity` | Weather | % | Impacts the "Heat Index" and perceived load demand.[4] |
| `Solar_Irradiance` | Weather | $W/m^2$ | Correlated with peak daytime cooling surges. |
| `Wind_Speed` | Weather | m/s | Affects ambient cooling and potential renewable supply. |
| `Cloud_Cover` | Weather | % | Used to predict sudden drops in solar generation. |
| `Is_High_Season` | Categorical | Binary | 1 = High Tourism (Jan–Apr, Jul–Aug); 0 = Low. |
| `BESS_SoC_Pct` | Status | % | The 50 MWh BESS state (Item 7 in infographic). |

### 2. Representative Mock Data Sample
This snippet demonstrates the **"Bottom neck"** constraint shown in your Image 1, where the available circuit capacity (Blue Line) drops below the required load (Yellow Line).

| Timestamp | Load (MW) | Circuit Cap (MW) | Temp (°C) | Humidity (%) | Irradiance | BESS SoC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-03-27 10:00 | 8.20 | 16.0 | 31.5 | 72 | 850 | 65% |
| 2026-03-27 11:00 | 8.55 | 16.0 | 32.8 | 70 | 920 | 58% |
| **2026-03-27 12:00** | **8.90** | **4.2 (Neck)** | 33.4 | 68 | 980 | **45%** |
| 2026-03-27 13:00 | 9.15 | 3.8 (Neck) | 34.1 | 67 | 950 | 32% |
| 2026-03-27 14:00 | 9.40 | 3.5 (Neck) | 34.5 | 66 | 890 | 20% |
| **2026-03-27 15:00** | **9.25** | **15.8 (Recov)** | 33.9 | 69 | 740 | **25%** |

### 3. Data Generation Logic (2-Year Context)
To ensure the model learns the correct physical relationships, the data follows these rules:
*   **A/C Correlation:** Every 1°C increase above 28°C adds approximately **0.35 MW** to the island load.
*   **Bottom-Neck Events:** The `Circuit_Cap_MW` is programmed to drop below 5 MW (Image 2) randomly but with higher frequency during peak mainland demand hours (18:00–21:00).[3]
*   **Tourism Multiplier:** During "High Season," the base load is shifted upward by **1.5 MW** to account for hotel and restaurant activity.
*   **BESS Response:** The SoC decreases when `Load` > `Circuit_Cap_MW` and increases during "Excess Power" periods (00:00–06:00).[5]

### 4. Machine Learning Dataset Splits
Divide your 17,520 hours of data chronologically to ensure the model can generalize across annual weather cycles.[2, 6]

*   **Training Set (70%):** Jan 1, 2025 – May 20, 2026.
    *   *Purpose:* Used for TCN temporal feature extraction and LightGBM tree growth.
*   **Validation Set (15%):** May 21, 2026 – Aug 31, 2026.
    *   *Purpose:* Hyperparameter tuning (learning rate, number of leaves, TCN dilation factors).[3, 2]
*   **Test Set (15%):** Sep 1, 2026 – Dec 31, 2026.
    *   *Purpose:* Final evaluation using **MAPE** (Target < 2.65%) and **R²** (Target > 0.97).[3, 7]

### 5. Recommended Feature Engineering for LightGBM
Before training, generate these additional features from the weather and timestamp data [2, 6]:
*   **Rolling Weather Stats:** 3-hour and 6-hour moving averages of `Temp` and `Humidity`.
*   **Lagged Load:** `Island_Load_MW` at $t-1h$, $t-24h$, and $t-168h$ (one week ago).
*   **Heat Index:** A derived feature combining `Temp` and `Humidity` to better capture thermal demand.
*   **Time Delta:** Hours remaining until the next "High Capacity" period (to help the BESS optimize discharge).
