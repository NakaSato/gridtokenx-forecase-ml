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
    curve = {float(k): v for k, v in dc["bsfc_curve"].items()}

    cap_mwh   = bc["capacity_mwh"]
    soc_min   = bc["soc_min"]
    soc_max   = bc["soc_max"]
    rated_mw  = dc["rated_mw"]
    opt_mw    = dc["optimal_output_mw"]   # 7.5 MW
    diesel_kg = oc["diesel_price_per_kg"]
    carbon_kg = oc["carbon_price_per_kg"]

    schedule = []
    soc = initial_soc

    for h, (load, circuit) in enumerate(zip(load_forecast, circuit_forecast)):
        delta = load - circuit   # positive = deficit, negative = surplus

        diesel_mw = 0.0
        bess_mw   = 0.0         # discharge (+) / charge (-)

        if delta <= 0:
            # Surplus: charge BESS
            charge = min(-delta, bc["charge_rate_mw"],
                         (soc_max - soc) * cap_mwh)
            bess_mw = -charge
        elif delta <= opt_mw:
            # BESS covers deficit alone
            discharge = min(delta, (soc - soc_min) * cap_mwh)
            bess_mw = discharge
            if discharge < delta:
                # BESS depleted — fire diesel for remainder
                remainder = delta - discharge
                diesel_mw = min(opt_mw, max(remainder, 0))
        else:
            # Large deficit: diesel at optimal + BESS for remainder
            diesel_mw = opt_mw
            net_after_diesel = delta - diesel_mw
            if net_after_diesel > 0:
                discharge = min(net_after_diesel, (soc - soc_min) * cap_mwh)
                bess_mw = discharge
            else:
                # Diesel overproduces → recharge BESS
                excess = -net_after_diesel
                charge = min(excess, bc["charge_rate_mw"],
                             (soc_max - soc) * cap_mwh)
                bess_mw = -charge

        # Update SoC
        soc = np.clip(soc - bess_mw / cap_mwh, soc_min, soc_max)

        # Fuel & carbon
        lf = diesel_mw / rated_mw if diesel_mw > 0 else 0.0
        sfc = _bsfc(lf, curve) if lf > 0 else 0.0          # g/kWh
        fuel_kg = sfc * diesel_mw / 1000.0                  # kg/h
        co2_kg  = fuel_kg * 2.68                            # diesel CO2 factor

        schedule.append(HourlyDispatch(
            hour=h, load_mw=round(load, 3), circuit_mw=round(circuit, 3),
            diesel_mw=round(diesel_mw, 3), bess_mw=round(bess_mw, 3),
            bess_soc=round(soc, 4), fuel_kg=round(fuel_kg, 3),
            carbon_kg=round(co2_kg, 3),
        ))

    return schedule


def schedule_summary(schedule: List[HourlyDispatch]) -> dict:
    total_fuel   = sum(s.fuel_kg   for s in schedule)
    total_carbon = sum(s.carbon_kg for s in schedule)
    diesel_hours = sum(1 for s in schedule if s.diesel_mw > 0)
    avg_diesel   = np.mean([s.diesel_mw for s in schedule])
    avg_bess     = np.mean([s.bess_mw for s in schedule if s.bess_mw > 0]) if any(s.bess_mw > 0 for s in schedule) else 0
    return {
        "total_fuel_kg":   round(total_fuel, 2),
        "total_carbon_kg": round(total_carbon, 2),
        "diesel_hours":    diesel_hours,
        "bess_soc_final":  schedule[-1].bess_soc,
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
