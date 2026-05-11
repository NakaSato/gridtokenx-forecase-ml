#!/bin/bash

# GridTokenX API Test Suite
# Requirements: curl, jq

API_URL="http://localhost:8000"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting GridTokenX API Tests...${NC}\n"

# 1. Health Check
echo -n "1. Testing /health... "
HEALTH=$(curl -s "$API_URL/health")
if [[ $HEALTH == *"ok"* ]]; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "$HEALTH"
fi

# 2. Metrics Check
echo -n "2. Testing /metrics... "
METRICS_HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/metrics")
if [[ $METRICS_HTTP_CODE -eq 200 ]]; then
    echo -e "${GREEN}PASS${NC}"
elif [[ $METRICS_HTTP_CODE -eq 404 ]]; then
    echo -e "${BLUE}SKIP (No evaluation report found)${NC}"
else
    echo -e "${RED}FAIL ($METRICS_HTTP_CODE)${NC}"
fi

# 3. Stream Telemetry
echo -n "3. Testing /stream/telemetry... "
TELEMETRY_JSON='{
  "row": {
    "phangan_load_mw": 18.5, "samui_load_mw": 65.2, "phangan_t2m": 28.1, "samui_t2m": 27.9,
    "t2m_celsius": 28.0, "rh_pct": 75.0, "ghi_w_m2": 450.0, "wind_ms": 3.2,
    "headroom_mw": 6.0, "max_capacity_mw": 13.3, "capacity_mw": 13.3, "bess_soc_pct": 65.0,
    "phangan_soc_pct": 70.0, "samui_soc_pct": 60.0, "tourist_index": 1.1,
    "hour_of_day": 14, "day_of_week": 2, "is_holiday": 0, "is_songkran": 0, "is_high_season": 1,
    "carbon_intensity": 440.0, "market_price": 75.0, "tao_load_mw": 7.2,
    "tao_load_mw_lag_1h": 7.1, "cable_flow_mw_lag_1h": 35.0, "kmb_flow_mw_lag_1h": 105.0,
    "tao_load_mw_lag_24h": 7.0, "cable_flow_mw_lag_24h": 34.5, "kmb_flow_mw_lag_24h": 104.0,
    "kmb_trend": 108.0, "kmb_seasonal": 0.5, "kmb_resid": 0.1
  },
  "samui_load_mw": 65.2,
  "phangan_load_mw": 18.5
}'

STREAM_RESP=$(curl -s -X POST "$API_URL/stream/telemetry" \
     -H "Content-Type: application/json" \
     -d "$TELEMETRY_JSON")

if [[ $STREAM_RESP == *"ingested"* ]] || [[ $STREAM_RESP == *"forecast"* ]]; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "$STREAM_RESP"
fi

# 4. Stream Actual
echo -n "4. Testing /stream/actual... "
ACTUAL_JSON='{
  "timestamp_iso": "2026-05-11T14:00:00Z",
  "actual_load_mw": 7.3,
  "forecast_load_mw": 7.1
}'
ACTUAL_RESP=$(curl -s -X POST "$API_URL/stream/actual" \
     -H "Content-Type: application/json" \
     -d "$ACTUAL_JSON")

if [[ $ACTUAL_RESP == *"metrics"* ]]; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "$ACTUAL_RESP"
fi

# 5. Grid Assets
echo -n "5. Testing /grid/assets... "
ASSETS_RESP=$(curl -s "$API_URL/grid/assets?table=koh_samui_grid&limit=1")
if [[ $ASSETS_RESP == *"FeatureCollection"* ]]; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
fi

# 6. Cluster Dispatch
echo -n "6. Testing /dispatch/cluster... "
CLUSTER_JSON='{
  "samui_load_mw": 70.0,
  "phangan_load_mw": 22.0,
  "tao_load_mw": 7.5
}'
CLUSTER_RESP=$(curl -s -X POST "$API_URL/dispatch/cluster" \
     -H "Content-Type: application/json" \
     -d "$CLUSTER_JSON")

if [[ $CLUSTER_RESP == *"dispatch"* ]]; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "$CLUSTER_RESP"
fi

# 7. Agent Explain
echo -n "7. Testing /agent/explain-dispatch... "
AGENT_JSON='{
  "optimized_schedule": {"hour_1": {"diesel": 2.5}},
  "baseline_schedule": {"hour_1": {"diesel": 3.0}}
}'
AGENT_RESP=$(curl -s -X POST "$API_URL/agent/explain-dispatch" \
     -H "Content-Type: application/json" \
     -d "$AGENT_JSON")

if [[ $AGENT_RESP == *"explanation"* ]]; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "$AGENT_RESP"
fi

echo -e "\n${BLUE}🏁 API Testing Complete.${NC}"
