# PEA Ground-Truth Validation Plan
**GridTokenX Commissioning Strategy for Ko Tao-Phangan-Samui**

## 🎯 Overview
This document outlines the formal validation protocol to be executed once proprietary PEA SCADA telemetry is released. The goal is to verify the high-fidelity AI Predictive Control Layer against historical "ground-truth" operations.

## 📊 Phase 1: Data Ingestion & Alignment
- **Telemetry Mapping:** Map PEA SCADA registers to the `TelemetryRow` schema.
- **Clock Synchronization:** Align all time-series data to hourly intervals (UTC+7).
- **Climate Verification:** Cross-reference PEA load spikes with ERA5-Land weather telemetry to finalize the Heat Index coefficients.

## 🧪 Phase 2: Predictive Accuracy Audit
- **Backtest MAPE:** Target < 2.65% on the last 12 months of historical data.
- **Residual Analysis:** Identify specific periods where the model deviates from PEA actuals (e.g., during unrecorded local festivals).

## 💰 Phase 3: Economic Viability Verification
- **Shadow Dispatch:** Run the ISCA optimizer against historical records.
- **KPIs:**
    - **Fuel Savings:** Target > 22.0% reduction in Diesel Fuel consumption.
    - **Asset Health:** Confirm BESS SoC compliance within 20-80% bounds.

## 🛡️ Phase 4: Resilience & Incident Playback
- **Bottleneck Simulation:** Play back historical 115 kV cable incidents.
- **Early Warning:** Verify that the system provides > 6h lead time for dispatch adjustments.
- **Cluster ADMM:** Test decentralized power-sharing during localized node failures.

## 📋 Phase 5: Final Commissioning Report
- **Executive Summary:** Comparative analysis of AI vs. Legacy Dispatch.
- **ROI Proof:** Levelized Cost of Energy (LCOE) reduction report.
- **Edge Deployment:** Final hardware commissioning on Ko Tao edge controllers.
