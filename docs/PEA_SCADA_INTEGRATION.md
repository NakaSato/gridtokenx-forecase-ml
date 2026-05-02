# PEA SCADA Endpoint Integration Plan
**GridTokenX — Ko Tao-Phangan-Samui Predictive Control Layer**
**Status:** Awaiting PEA data release | **Owner:** GridTokenX Engineering

---

## 1. Overview

This document defines the integration plan between the GridTokenX AI Predictive Control system and the PEA SCADA telemetry endpoint for the Ko Samui–Ko Phangan–Ko Tao islanded microgrid cluster.

All model training, validation, and commissioning infrastructure is **complete and ready**. The only remaining step is connecting to the live PEA data feed.

---

## 2. Pre-Integration Checklist (Ask PEA First)

Before writing integration code, confirm the following with PEA Smart Grid Innovation (`sgi@pea.co.th`):

| # | Question | Why It Matters |
| :- | :--- | :--- |
| 1 | **Protocol** — REST push, REST pull, MQTT, or SFTP batch? | Determines which adapter to activate |
| 2 | **Cadence** — 1-min, 5-min, or hourly intervals? | Determines buffer aggregation logic |
| 3 | **Auth** — API key, OAuth2, IP whitelist, or VPN? | Required before any connection attempt |
| 4 | **Available registers** — load MW, BESS SoC, circuit cap, temperature, humidity? | Missing registers fall back to synthetic defaults |
| 5 | **Historical backfill** — minimum 3 months available? | Required for `just pea-onboard` calibration step |
| 6 | **Endpoint URL** — production + staging? | Staging needed for dry-run validation |

### Draft Request to PEA

> Subject: GridTokenX — SCADA Integration Request for Ko Samui/Tao/Phangan
>
> We are ready to integrate the GridTokenX AI Predictive Control system with the PEA SCADA endpoint for the Ko Samui–Phangan–Tao microgrid cluster. Our onboarding pipeline is complete and tested.
>
> Please confirm:
> 1. Data delivery method (REST API / MQTT / SFTP)
> 2. Update frequency (1 min / 5 min / hourly)
> 3. Authentication method and test credentials
> 4. Available registers: Island Load MW, BESS SoC %, Circuit Capacity MW, Dry Bulb Temp, Relative Humidity
> 5. Historical backfill availability (minimum 3 months for model calibration)
> 6. Staging/test endpoint URL
>
> Our system is deployed on ARM64 edge controllers at Ko Tao and can receive or pull data in any of the above formats.

---

## 3. Integration Architecture

```
PEA SCADA
    │
    ├─[REST pull]──┐
    ├─[MQTT sub]───┤
    └─[SFTP batch]─┤
                   ▼
         api/scada_ingest.py          ← protocol adapter (mode-switchable)
                   │
                   ▼
         POST /stream/telemetry       ← existing FastAPI endpoint
                   │
          ┌────────┴────────┐
          ▼                 ▼
    TCN+LGBM forecast   MILP optimizer
          │                 │
          └────────┬────────┘
                   ▼
         dispatch schedule + early warnings
                   │
          ┌────────┴────────┐
          ▼                 ▼
    PEA SCADA write-back   results/live_metrics.json
    (optional, if PEA      (local archive)
     supports it)
```

---

## 4. Protocol Modes

### Mode A — REST Pull (most likely for new PEA deployments)

```
GET https://pea-scada.local/api/telemetry/latest
Authorization: Bearer <SCADA_API_KEY>

Response:
{
  "timestamp": "2026-05-01T10:00:00+07:00",
  "PEA_LOAD_MW": 7.42,
  "CABLE_CAP_MW": 12.0,
  "BESS_SOC": 68.5,
  "TEMP_DRY": 31.2,
  "HUMIDITY": 78.0
}
```

Polling interval: match SCADA cadence (5-min or hourly).
Aggregation: if cadence < 1h, buffer and average to hourly before inference.

### Mode B — MQTT Subscribe (common for real-time BESS monitoring)

```
Broker:  mqtt://pea-scada.local:1883
Topic:   pea/kotao/telemetry
QoS:     1
Auth:    username/password or TLS client cert
```

### Mode C — SFTP Batch (legacy PEA provincial offices)

```
Host:    sftp.pea.co.th
Path:    /export/kotao/telemetry_YYYYMMDD.csv
Cadence: daily drop at 01:00 UTC+7
Format:  CSV with PEA register names (mapped via PEA_COLUMN_MAP)
```

---

## 5. Column Mapping

PEA SCADA register names → GridTokenX schema (defined in `data/pea_onboard.py`):

| PEA Register | GridTokenX Column | Required |
| :--- | :--- | :--- |
| `PEA_LOAD_MW` | `Island_Load_MW` | ✅ Yes |
| `TEMP_DRY` | `Dry_Bulb_Temp` | ✅ Yes |
| `HUMIDITY` | `Rel_Humidity` | ✅ Yes |
| `CABLE_CAP_MW` | `Circuit_Cap_MW` | ⚠️ Default: config max |
| `BESS_SOC` | `BESS_SoC_Pct` | ⚠️ Default: 65% |
| `SOLAR_W_M2` | `Solar_Irradiance` | ⚠️ Default: 200 W/m² |
| `CARBON_INT` | `Carbon_Intensity` | ⚠️ Default: 0.5 |
| `MARKET_PRICE_THB` | `Market_Price` | ⚠️ Default: 3.5 |
| `TOURIST_IDX` | `Tourist_Index` | ⚠️ Default: 0.7 |

Update `PEA_COLUMN_MAP` at the top of `data/pea_onboard.py` once actual register names are confirmed.

---

## 6. Activation Sequence

Once PEA confirms protocol and credentials:

```bash
# Step 1 — Set credentials on edge device
nano /opt/gridtokenx/edge.env
# Set: SCADA_PUSH_URL, SCADA_API_KEY, SCADA_MODE

# Step 2 — Dry-run (no model writes, logs only)
just scada-ingest --dry-run

# Step 3 — Collect 3 months of live data, then onboard
just pea-onboard data/raw/pea_telemetry_raw.csv 3

# Step 4 — Verify model performance on real data
just pea-backtest data/raw/pea_telemetry_raw.csv

# Step 5 — Go live
just scada-ingest
sudo systemctl restart gridtokenx
```

---

## 7. Files to Build (on protocol confirmation)

| File | Status | Description |
| :--- | :--- | :--- |
| `api/scada_ingest.py` | 🔲 Pending protocol | Protocol adapter — REST/MQTT/SFTP → `/stream/telemetry` |
| `data/pea_onboard.py` | ✅ Ready | Batch calibration pipeline |
| `deploy/edge.env.example` | ✅ Ready | Credentials template |
| `justfile` `scada-ingest` | 🔲 Pending protocol | Daemon start recipe |

---

## 8. Data Archive Strategy

All raw PEA telemetry is appended to `data/raw_pea/` before any transformation:

```
data/raw_pea/
  YYYY-MM-DD.parquet    ← daily files, immutable after write
  index.json            ← manifest: date range, row count, checksum
```

This ensures the original PEA data is always recoverable regardless of pipeline changes.

---

## 9. Fallback Behaviour

If the SCADA connection drops, the edge controller falls back gracefully:

| Condition | Behaviour |
| :--- | :--- |
| SCADA timeout < 1h | Use last known telemetry + flag in response |
| SCADA timeout 1–6h | Switch to synthetic forecast (ERA5 weather + tourist calendar) |
| SCADA timeout > 6h | Early warning alert; hold last dispatch schedule |
| BESS SoC unavailable | Assume 65% (conservative mid-range) |
| Circuit cap unavailable | Assume config max (16 MW) — conservative |

---

## 10. Success Criteria

Integration is considered complete when:

- [ ] Live MAPE on PEA actuals ≤ 2.65% over 30-day rolling window
- [ ] Fuel savings ≥ 22% vs PEA legacy dispatch (confirmed by PEA operations log)
- [ ] Zero unplanned outages attributable to AI dispatch over 90-day trial
- [ ] `results/pea_onboard_report.json` generated with real data
- [ ] PEA operations team sign-off on commissioning report

---

*Last updated: 2026-05-01 | Next action: Send pre-integration checklist to PEA SGI*