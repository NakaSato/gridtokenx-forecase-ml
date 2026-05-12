"""
Improved Sine-Cosine Algorithm (ISCA) for dispatch cost minimization.
Minimizes: C_total = C_diesel + C_bess_degradation + C_carbon

Decision variables per hour (24h horizon):
  x[0:24]  = diesel on/off (binary, relaxed to [0,1] then rounded)
  x[24:48] = BESS dispatch MW (negative=charge, positive=discharge)
"""
import numpy as np
import yaml
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from domain.dispatch import _bsfc
from domain.entities import HourlyDispatch
from typing import List


def _cost(x: np.ndarray, load: np.ndarray, circuit: np.ndarray,
          cfg: dict, initial_soc: float) -> float:
    bc = cfg["bess"]; dc = cfg["diesel"]; oc = cfg["optimizer"]
    cap = bc["capacity_mwh"]; soc_min = bc["soc_min"]; soc_max = bc["soc_max"]
    rated = dc["rated_mw"]
    unit_rating = dc.get("unit_rating_mw", 2.0)
    p_min_unit = dc.get("min_load_mw", 0.5)
    
    curve = {float(k): v for k, v in dc["bsfc_curve"].items()}
    d_price = oc["diesel_price_per_kg"]
    c_price = oc["carbon_price_per_kg"]
    deg_cost = oc["bess_degradation_cost_per_cycle"]

    T = len(load)
    # Decision variables: x[:T] = total_diesel_mw
    total_diesel_mw = x[:T].copy()
    
    # Unit Commitment Logic inside _cost
    # If total_diesel_mw > 0, we need at least one unit.
    # We assume units are dispatched optimally to share the load.
    units_active = np.ceil(total_diesel_mw / unit_rating).astype(int)
    # Penalize if total_diesel_mw < units_active * p_min_unit
    
    bess_mw   = x[T:].clip(-bc["charge_rate_mw"] if cap > 0 else 0,
                              (soc_max - soc_min) * cap if cap > 0 else 0)

    soc = initial_soc
    total = 0.0
    penalty = 0.0
    dt = 24 / (T + 1e-6)

    for h in range(T):
        d_mw = total_diesel_mw[h]
        u_act = units_active[h]
        b_mw = bess_mw[h]

        # Load balance penalty
        supplied = circuit[h] + d_mw + b_mw
        imbalance = max(0, load[h] - supplied) 
        penalty += imbalance * 1000000.0 # High penalty for deficit
        
        # Penalize over-generation (spill) if no BESS to absorb
        spill = max(0, supplied - load[h])
        if cap == 0:
            penalty += spill * 10000.0

        # Min load penalty per unit
        if u_act > 0:
            if d_mw < u_act * p_min_unit:
                penalty += (u_act * p_min_unit - d_mw) * 1000.0

        # SoC update & bounds penalty
        if cap > 0:
            new_soc = soc - b_mw / cap
            if new_soc < soc_min:
                penalty += (soc_min - new_soc) * 500.0
            if new_soc > soc_max:
                penalty += (new_soc - soc_max) * 500.0
            soc = np.clip(new_soc, soc_min, soc_max)
        else:
            if abs(b_mw) > 1e-6:
                penalty += abs(b_mw) * 1000.0 
            soc = initial_soc

        # Fuel cost (Unit-Aware)
        if u_act > 0 and d_mw > 0:
            lf = (d_mw / u_act) / unit_rating
            sfc = _bsfc(lf, curve)
            fuel_kg = sfc * d_mw * dt
            co2_kg  = fuel_kg * 2.68
            total += fuel_kg * d_price + co2_kg * c_price

    # BESS degradation: count discharge cycles (simplified)
    if cap > 0:
        discharge_kwh = np.sum(bess_mw.clip(0)) * dt
        cycles = discharge_kwh / (cap * (soc_max - soc_min))
        total += cycles * deg_cost

    return total + penalty


def isca_optimize(
    load: np.ndarray,
    circuit: np.ndarray,
    initial_soc: float = 0.65,
    cfg: dict = None,
) -> dict:
    if cfg is None:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

    ic = cfg["optimizer"]["isca"]
    pop_size  = ic["population"]
    max_iter  = ic["max_iter"]
    elite_k   = ic["elite_top_k"]
    bc = cfg["bess"]; dc = cfg["diesel"]
    cap = bc["capacity_mwh"]
    rated = dc["rated_mw"]
    unit_rating = dc.get("unit_rating_mw", 2.0)
    p_min_unit = dc.get("min_load_mw", 0.5)

    T = len(load)
    rng = np.random.default_rng(42)
    dim = 2 * T 

    # Bounds
    lb = np.concatenate([np.zeros(T),
                         np.full(T, -bc["charge_rate_mw"] if cap > 0 else 0.0)])
    ub = np.concatenate([np.full(T, rated),
                         np.full(T, (bc["soc_max"] - bc["soc_min"]) * cap if cap > 0 else 0.0)])

    # Initialize population
    pop  = lb + rng.random((pop_size, dim)) * (ub - lb)
    for i in range(pop_size):
        for h in range(T):
            deficit = max(0, load[h] - circuit[h])
            if deficit > 0:
                # Initialize with slightly more than deficit to encourage finding valid points
                pop[i, h] = np.clip(deficit * rng.uniform(1.0, 1.1), 0, rated)
    
    fits = np.array([_cost(p, load, circuit, cfg, initial_soc) for p in pop])

    best_idx = np.argmin(fits)
    best_x   = pop[best_idx].copy()
    best_fit = fits[best_idx]

    for t in range(1, max_iter + 1):
        r1 = 2 - t * (2 / max_iter)   # decreasing inertia
        r2 = 2 * np.pi * rng.random((pop_size, dim))
        r3 = rng.random((pop_size, dim))
        r4 = rng.random((pop_size, dim))

        pop = np.where(r4 < 0.5, 
                       pop + r1 * np.sin(r2) * np.abs(r3 * best_x - pop), 
                       pop + r1 * np.cos(r2) * np.abs(r3 * best_x - pop))
        pop = np.clip(pop, lb, ub)

        new_fits  = np.array([_cost(p, load, circuit, cfg, initial_soc) for p in pop])
        
        # Elite retention
        elite_idx = np.argsort(fits)[:elite_k]
        for ei in elite_idx:
            worst = np.argmax(new_fits)
            if fits[ei] < new_fits[worst]:
                pop[worst]      = pop[ei].copy()
                new_fits[worst] = fits[ei]

        fits = new_fits
        cur_best = np.argmin(fits)
        if fits[cur_best] < best_fit:
            best_fit = fits[cur_best]
            best_x   = pop[cur_best].copy()
            
        if t % 50 == 0:
            print(f"      [ISCA] Iter {t}/{max_iter} | Best Fit: {best_fit:.2f}")

    # Decode
    diesel_mw = best_x[:T].copy()
    units_active = np.ceil(diesel_mw / unit_rating).astype(int)
    
    # Final cleanup of decoding (similar to _cost but final)
    for h in range(T):
        if diesel_mw[h] < units_active[h] * p_min_unit:
            diesel_mw[h] = units_active[h] * p_min_unit
            
    bess_mw   = best_x[T:].clip(-bc["charge_rate_mw"] if cap > 0 else 0,
                                  (bc["soc_max"] - bc["soc_min"]) * cap if cap > 0 else 0)
    
    curve  = {float(k): v for k, v in dc["bsfc_curve"].items()}
    schedule, soc = [], initial_soc
    total_fuel = total_carbon = 0.0
    dt = 24 / (T + 1e-6)
    
    for h in range(T):
        d_mw = diesel_mw[h]
        u_act = units_active[h]
        b_mw = bess_mw[h]
        if cap > 0:
            soc = np.clip(soc - b_mw / cap, bc["soc_min"], bc["soc_max"])
        else:
            soc = initial_soc
            
        if u_act > 0:
            lf = (d_mw / u_act) / unit_rating
            sfc = _bsfc(lf, curve)
            fuel_kg = sfc * d_mw * dt
            co2_kg  = fuel_kg * 2.68
        else:
            fuel_kg = co2_kg = 0.0

        total_fuel    += fuel_kg
        total_carbon  += co2_kg
        schedule.append(HourlyDispatch(
            hour=h, load_mw=round(load[h], 3), circuit_mw=round(circuit[h], 3),
            diesel_mw=round(d_mw, 3), bess_mw=round(b_mw, 3),
            bess_soc=round(soc, 4), fuel_kg=round(fuel_kg, 3),
            carbon_kg=round(co2_kg, 3),
        ))

    return {
        "schedule":       schedule,
        "best_cost":      round(best_fit, 4),
        "total_fuel_kg":  round(total_fuel, 2),
        "total_carbon_kg": round(total_carbon, 2),
        "diesel_hours":   int(np.sum(diesel_mw > 0)),
        "bess_soc_final": round(soc, 4),
    }


if __name__ == "__main__":
    import json
    rng = np.random.default_rng(1)
    load    = rng.uniform(6, 10, 24)
    circuit = np.where(np.isin(np.arange(24), [18,19,20,21]),
                       rng.uniform(1, 4, 24), rng.uniform(12, 16, 24))
    result = isca_optimize(load, circuit)
    print(json.dumps({k: v for k, v in result.items() if k != "schedule"}, indent=2))
    for s in result["schedule"]:
        print(f"h{s.hour:02d} diesel={s.diesel_mw:.1f} bess={s.bess_mw:+.2f} "
              f"soc={s.bess_soc:.2f} fuel={s.fuel_kg:.2f}kg co2={s.carbon_kg:.2f}kg")
