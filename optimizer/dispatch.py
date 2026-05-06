"""
Rule-based predictive dispatch using BSFC curve.
Given 24h forecast arrays, returns hourly dispatch schedule.
"""
import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class HourlyDispatch:
    hour: int
    load_mw: float
    circuit_mw: float
    diesel_mw: float
    bess_mw: float        # positive = discharge, negative = charge
    bess_soc: float       # 0–1
    fuel_kg: float
    carbon_kg: float


def _bsfc(load_factor: float, curve: dict) -> float:
    """Interpolate BSFC (g/kWh) from load factor."""
    keys = sorted(float(k) for k in curve)
    vals = [curve[str(k)] if str(k) in curve else curve[k] for k in keys]
    return float(np.interp(load_factor, keys, vals))


def run_dispatch(
    load_forecast: np.ndarray,
    circuit_forecast: np.ndarray,
    initial_soc: float = 0.65,
    cfg: dict = None,
) -> List[HourlyDispatch]:
    if cfg is None:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

    bc = cfg["bess"]
    dc = cfg["diesel"]
    oc = cfg["optimizer"]
    freq = cfg["data"].get("frequency", "h")
    sph = 4 if freq == "15min" else 1
    dt = 1.0 / sph
    curve = {float(k): v for k, v in dc["bsfc_curve"].items()}

    cap_mwh   = max(1e-6, bc["capacity_mwh"])
    soc_min   = bc["soc_min"]
    soc_max   = bc["soc_max"]
    
    # 5 x 2MW Configuration
    num_units = 5
    unit_rating = 2.0
    p_min_unit = 0.5 # Min load per 2MW unit

    schedule = []
    soc = initial_soc

    for h, (load, circuit) in enumerate(zip(load_forecast, circuit_forecast)):
        delta = load - circuit   # positive = deficit

        diesel_mw = 0.0
        bess_mw   = 0.0
        units_active = 0

        if delta <= 0:
            # Surplus: charge BESS
            charge = min(-delta, bc["charge_rate_mw"], (soc_max - soc) * cap_mwh / dt)
            bess_mw = -charge
        else:
            # Deficit: Lowest Cost Priority (Diesel vs BESS)
            bess_avail_mw = min(bc["charge_rate_mw"], (soc - soc_min) * cap_mwh / dt)
            
            # 1. Calculate marginal BESS cost
            depth = soc_max - soc_min
            deg_per_mwh = oc["bess_degradation_cost_per_cycle"] / (cap_mwh * depth) if cap_mwh > 0 else 9999.0
            cost_bess_per_mw = deg_per_mwh * dt
            
            # 2. Calculate marginal Diesel cost
            # Assume 1 unit running at required load (or p_min_unit if small)
            test_diesel_mw = max(delta, p_min_unit)
            lf = test_diesel_mw / unit_rating
            sfc = _bsfc(lf, curve)
            fuel_kg_per_h = sfc * test_diesel_mw
            cost_per_kg = oc["diesel_price_per_kg"] + 2.68 * oc["carbon_price_per_kg"]
            cost_diesel_per_mw = (fuel_kg_per_h * cost_per_kg * dt) / test_diesel_mw

            if cost_bess_per_mw <= cost_diesel_per_mw and bess_avail_mw > 0:
                # BESS is cheaper or Diesel is off
                bess_mw = min(delta, bess_avail_mw)
                diesel_mw = delta - bess_mw
                
                # Check minimum diesel load constraint
                if 0 < diesel_mw < p_min_unit:
                    # If remaining is too small for diesel, try to push more to BESS if capable
                    # Otherwise, force diesel to min load and charge BESS with excess
                    diesel_mw = p_min_unit
                    excess = diesel_mw - (delta - bess_mw)
                    bess_mw -= excess
            else:
                # Diesel is cheaper
                units_active = int(np.ceil(delta / unit_rating))
                units_active = min(units_active, num_units)
                max_gen = units_active * unit_rating
                
                diesel_mw = min(delta, max_gen)
                bess_mw = delta - diesel_mw
                
                # Minimum load constraint
                if diesel_mw > 0 and diesel_mw < units_active * p_min_unit:
                    diesel_mw = units_active * p_min_unit
                    bess_mw = delta - diesel_mw
                    
                # BESS bounds check if BESS is forced to discharge
                if bess_mw > bess_avail_mw:
                    bess_mw = bess_avail_mw
                    diesel_mw = delta - bess_mw
                    
            units_active = int(np.ceil(diesel_mw / unit_rating)) if diesel_mw > 0 else 0

        # Update SoC
        soc = np.clip(soc - (bess_mw * dt) / cap_mwh, soc_min, soc_max)

        # Fuel (Unit-aware BSFC)
        fuel_kg = 0.0
        if units_active > 0 and diesel_mw > 0:
            lf = (diesel_mw / units_active) / unit_rating
            sfc = _bsfc(lf, curve)
            fuel_kg = (sfc * diesel_mw) * dt

        schedule.append(HourlyDispatch(
            hour=h, load_mw=round(load, 3), circuit_mw=round(circuit, 3),
            diesel_mw=round(diesel_mw, 3), bess_mw=round(bess_mw, 3),
            bess_soc=round(soc, 4), fuel_kg=round(fuel_kg, 3),
            carbon_kg=round(fuel_kg * 2.68, 3),
        ))

    return schedule


def schedule_summary(schedule: List[HourlyDispatch]) -> dict:
    if not schedule: return {}
    total_fuel = sum(s.fuel_kg for s in schedule)
    unit_rating = 2.0
    step_plan = [int(np.ceil(s.diesel_mw / unit_rating)) if s.diesel_mw > 0 else 0 for s in schedule]
    
    return {
        "total_fuel_kg": round(total_fuel, 2),
        "diesel_hours": sum(1 for s in schedule if s.diesel_mw > 0),
        "max_units_active": max(step_plan) if step_plan else 0,
        "step_plan_units": step_plan,
        "bess_soc_final": schedule[-1].bess_soc,
    }


if __name__ == "__main__":
    import json
    rng = np.random.default_rng(0)
    load     = rng.uniform(6, 10, 24)
    circuit  = np.where(np.isin(np.arange(24), [18,19,20,21]),
                        rng.uniform(1, 4, 24), rng.uniform(12, 16, 24))
    sched = run_dispatch(load, circuit)
    summary = schedule_summary(sched)
    print(json.dumps(summary, indent=2))
    for s in sched:
        print(f"h{s.hour:02d} load={s.load_mw:.2f} circ={s.circuit_mw:.2f} "
              f"diesel={s.diesel_mw:.2f} bess={s.bess_mw:+.2f} "
              f"soc={s.bess_soc:.2f} fuel={s.fuel_kg:.2f}kg")
