"""
Early Warning System — detects impending BESS depletion / bottleneck events.

Logic:
  - Simulate BESS SoC forward using forecast load & circuit capacity
  - Alert if SoC will drop below threshold within lookahead_hours
  - Alert if Net_Delta will exceed BESS dispatchable capacity
  - Run pandapower AC power flow to get physics-accurate line loading %
    and use it as the bottleneck signal (replaces simple circuit_forecast < 5 MW threshold)

Can be run standalone or imported by the API.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Warning:
    level: str          # "CRITICAL" | "WARNING" | "INFO"
    hour:  int          # hours from now when event occurs
    message: str
    soc_at_event: float
    hvdc_loading_pct: Optional[float] = None   # pandapower result, None if not run


def _run_physics_check(
    tao_load_mw: float,
    phangan_load_mw: float,
    samui_load_mw: float,
) -> dict:
    """
    Run pandapower AC power flow for the given island loads.
    Returns HVDC connector loading % and voltage at Ko Tao.
    Falls back to None on import error (pandapower optional at runtime).
    """
    try:
        from research.pandapower_model import verify_dispatch_stability
        return verify_dispatch_stability(
            tao_load_mw=tao_load_mw,
            phangan_load_mw=phangan_load_mw,
            samui_load_mw=samui_load_mw,
        )
    except Exception:
        return {}


def check_warnings(
    load_forecast: np.ndarray,    # 24h ahead (Ko Tao)
    circuit_forecast: np.ndarray, # 24h ahead
    current_soc: float,           # 0–1
    cfg: dict = None,
    lookahead_hours: int = 6,
    # Optional cluster loads for pandapower physics check
    phangan_forecast: Optional[np.ndarray] = None,
    samui_forecast: Optional[np.ndarray] = None,
) -> List[Warning]:
    if cfg is None:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

    cc  = cfg["cluster"]
    ca  = cc["assets"]
    bc  = cfg["bess"]
    dc  = cfg["diesel"]
    
    # N-1 Limits from coordinated config
    n1_critical = cc["admm"]["mainland_n1_critical_mw"]
    n1_alert    = cc["admm"]["mainland_n1_alert_mw"]
    
    # BESS Capacity - use Ko Tao specific if standalone, or Cluster BESS if provided
    # Ko Tao has no local BESS, so stand-alone check focuses on Import Capacity
    cap = bc["capacity_mwh"]
    if phangan_forecast is not None and samui_forecast is not None:
        # Account for Samui BESS (Node 7) in cluster warning mode
        cap = ca["ko_samui"]["bess_mwh"]
        
    soc_min   = bc["soc_min"]
    soc_warn  = soc_min + 0.10   # warn at 30%
    opt_mw    = dc["optimal_output_mw"]

    warnings = []
    soc = current_soc

    for h in range(min(lookahead_hours, len(load_forecast))):
        delta = load_forecast[h] - circuit_forecast[h]

        # ── Cluster-wide N-1 Contingency Check ──────────────────────────────
        if phangan_forecast is not None and samui_forecast is not None:
            total_cluster_load = float(load_forecast[h] + phangan_forecast[h] + samui_forecast[h])
            
            if total_cluster_load >= n1_critical:
                warnings.append(Warning(
                    level="CRITICAL", hour=h,
                    message=f"Total Cluster Load {total_cluster_load:.1f} MW exceeds N-1 limit ({n1_critical} MW). "
                            "Single cable failure will cause blackout.",
                    soc_at_event=round(soc, 3)
                ))
            elif total_cluster_load >= n1_alert:
                warnings.append(Warning(
                    level="WARNING", hour=h,
                    message=f"Total Cluster Load {total_cluster_load:.1f} MW approaching N-1 limit. "
                            "Stage island generation to reduce mainland intake.",
                    soc_at_event=round(soc, 3)
                ))

        # Project SoC without diesel intervention
        if cap > 0:
            if delta > 0:
                soc = max(soc_min, soc - delta / cap)
            else:
                soc = min(bc["soc_max"], soc + (-delta) / cap)
        else:
            # No BESS case: focus on import limit
            if delta > 0:
                soc = soc_min # Virtual depletion clipped to min for test coordination

        dispatchable_mwh = (soc - soc_min) * cap if cap > 0 else 0.0

        # ── Physics check via pandapower ──────────────────────────────────────
        hvdc_pct = None
        v_tao    = None
        if phangan_forecast is not None and samui_forecast is not None:
            ph = _run_physics_check(
                tao_load_mw=float(load_forecast[h]),
                phangan_load_mw=float(phangan_forecast[h]),
                samui_load_mw=float(samui_forecast[h]),
            )
            hvdc_pct = ph.get("bottleneck_loading_pct")
            v_tao    = ph.get("v_tao_pu")

            # CRITICAL: physics overload on HVDC connector
            if hvdc_pct is not None and hvdc_pct > 100.0:
                warnings.append(Warning(
                    level="CRITICAL", hour=h,
                    message=f"HVDC Koh Samui Connector at {hvdc_pct:.1f}% loading at h+{h}. "
                            f"Cable thermal limit exceeded — curtailment required.",
                    soc_at_event=round(soc, 3),
                    hvdc_loading_pct=round(hvdc_pct, 1),
                ))
            elif hvdc_pct is not None and hvdc_pct > 85.0:
                warnings.append(Warning(
                    level="WARNING", hour=h,
                    message=f"HVDC Connector approaching limit: {hvdc_pct:.1f}% at h+{h}. "
                            f"Pre-charge BESS and stage diesel.",
                    soc_at_event=round(soc, 3),
                    hvdc_loading_pct=round(hvdc_pct, 1),
                ))

            # WARNING: voltage violation at Ko Tao
            if v_tao is not None and not (0.95 <= v_tao <= 1.05):
                warnings.append(Warning(
                    level="WARNING", hour=h,
                    message=f"Ko Tao voltage {v_tao:.4f} p.u. outside 0.95–1.05 band at h+{h}.",
                    soc_at_event=round(soc, 3),
                    hvdc_loading_pct=hvdc_pct,
                ))

        # ── BESS / diesel checks (unchanged) ─────────────────────────────────
        if soc <= soc_min and delta > opt_mw:
            warnings.append(Warning(
                level="CRITICAL", hour=h,
                message=f"BESS depleted at h+{h} and deficit {delta:.1f} MW exceeds diesel capacity. Load shedding risk.",
                soc_at_event=round(soc, 3),
            ))
        elif soc <= soc_min:
            warnings.append(Warning(
                level="CRITICAL", hour=h,
                message=f"BESS will reach minimum SoC at h+{h}. Start diesel pre-emptively.",
                soc_at_event=round(soc, 3),
            ))
        elif soc <= soc_warn:
            warnings.append(Warning(
                level="WARNING", hour=h,
                message=f"BESS SoC at {soc*100:.1f}% at h+{h}. Consider starting diesel.",
                soc_at_event=round(soc, 3),
            ))

        # Fallback bottleneck check (no pandapower): use circuit_forecast threshold
        if hvdc_pct is None and circuit_forecast[h] < 5.0 and dispatchable_mwh < delta * 2:
            warnings.append(Warning(
                level="WARNING", hour=h,
                message=f"Bottleneck at h+{h}: circuit={circuit_forecast[h]:.1f} MW, "
                        f"BESS reserve={dispatchable_mwh:.1f} MWh covers only {dispatchable_mwh/max(delta,0.1):.1f}h.",
                soc_at_event=round(soc, 3),
            ))

    # INFO: upcoming bottleneck window (circuit-based fallback)
    if phangan_forecast is None:
        bottleneck_hours = [h for h in range(len(circuit_forecast)) if circuit_forecast[h] < 5.0]
        if bottleneck_hours:
            first = bottleneck_hours[0]
            warnings.append(Warning(
                level="INFO", hour=first,
                message=f"Bottleneck window detected at h+{first} "
                        f"({len(bottleneck_hours)} hours, min={min(circuit_forecast[bottleneck_hours]):.1f} MW). "
                        f"Charge BESS now if possible.",
                soc_at_event=round(current_soc, 3),
            ))

    return warnings


def format_warnings(warnings: List[Warning]) -> str:
    if not warnings:
        return "✅ No warnings — grid stable for next forecast window."
    icons = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}
    lines = []
    for w in sorted(warnings, key=lambda x: (x.level != "CRITICAL", x.hour)):
        lines.append(f"{icons[w.level]} [{w.level}] h+{w.hour:02d} | SoC={w.soc_at_event*100:.1f}% | {w.message}")
    return "\n".join(lines)


if __name__ == "__main__":
    import pandas as pd

    test = pd.read_parquet("data/processed/test.parquet")
    # Simulate: use actual values as "forecast" for a bottleneck window
    # Find first bottleneck day
    circuit = test["Circuit_Cap_MW"].values
    load    = test["Island_Load_MW"].values
    bottleneck_idx = next((i for i in range(len(circuit)-24) if circuit[i] < 5), 0)

    load_fc    = load[bottleneck_idx: bottleneck_idx + 24]
    circuit_fc = circuit[bottleneck_idx: bottleneck_idx + 24]
    current_soc = 0.45  # low SoC scenario

    warnings = check_warnings(load_fc, circuit_fc, current_soc, lookahead_hours=6)
    print(format_warnings(warnings))
