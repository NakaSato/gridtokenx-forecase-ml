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
from optimizer.dispatch import _bsfc, HourlyDispatch
from typing import List


def _cost(x: np.ndarray, load: np.ndarray, circuit: np.ndarray,
          cfg: dict, initial_soc: float) -> float:
    bc = cfg["bess"]; dc = cfg["diesel"]; oc = cfg["optimizer"]
    cap = bc["capacity_mwh"]; soc_min = bc["soc_min"]; soc_max = bc["soc_max"]
    rated = dc["rated_mw"]; opt_mw = dc["optimal_output_mw"]
    curve = {float(k): v for k, v in dc["bsfc_curve"].items()}
    d_price = oc["diesel_price_per_kg"]
    c_price = oc["carbon_price_per_kg"]
    deg_cost = oc["bess_degradation_cost_per_cycle"]

    diesel_on = np.round(x[:24].clip(0, 1))   # binary
    bess_mw   = x[24:].clip(-bc["charge_rate_mw"],
                             (soc_max - soc_min) * cap)

    soc = initial_soc
    total = 0.0
    penalty = 0.0

    for h in range(24):
        d_mw = diesel_on[h] * opt_mw
        b_mw = bess_mw[h]

        # Load balance penalty
        supplied = circuit[h] + d_mw + b_mw
        imbalance = abs(supplied - load[h])
        penalty += imbalance * 1000.0

        # SoC update & bounds penalty
        new_soc = soc - b_mw / cap
        if new_soc < soc_min:
            penalty += (soc_min - new_soc) * 500.0
        if new_soc > soc_max:
            penalty += (new_soc - soc_max) * 500.0
        soc = np.clip(new_soc, soc_min, soc_max)

        # Fuel cost
        lf = d_mw / rated if d_mw > 0 else 0.0
        sfc = _bsfc(lf, curve) if lf > 0 else 0.0
        fuel_kg = sfc * d_mw / 1000.0
        co2_kg  = fuel_kg * 2.68
        total += fuel_kg * d_price + co2_kg * c_price

    # BESS degradation: count discharge cycles (simplified)
    discharge_kwh = np.sum(bess_mw.clip(0) )
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

    rng = np.random.default_rng(42)
    dim = 48  # 24 diesel + 24 bess

    # Bounds
    lb = np.concatenate([np.zeros(24),
                         np.full(24, -bc["charge_rate_mw"])])
    ub = np.concatenate([np.ones(24),
                         np.full(24, (bc["soc_max"] - bc["soc_min"]) * cap)])

    # Initialize population
    pop  = lb + rng.random((pop_size, dim)) * (ub - lb)
    fits = np.array([_cost(p, load, circuit, cfg, initial_soc) for p in pop])

    best_idx = np.argmin(fits)
    best_x   = pop[best_idx].copy()
    best_fit = fits[best_idx]

    for t in range(1, max_iter + 1):
        r1 = 2 - t * (2 / max_iter)   # decreasing inertia
        r2 = 2 * np.pi * rng.random((pop_size, dim))
        r3 = rng.random((pop_size, dim))
        r4 = rng.random((pop_size, dim))

        # Sine-cosine position update toward best
        sine_move   = r1 * np.sin(r2) * np.abs(r3 * best_x - pop)
        cosine_move = r1 * np.cos(r2) * np.abs(r3 * best_x - pop)
        pop = np.where(r4 < 0.5, pop + sine_move, pop + cosine_move)
        pop = np.clip(pop, lb, ub)

        # Elite retention: keep top-k from previous generation
        elite_idx = np.argsort(fits)[:elite_k]
        new_fits  = np.array([_cost(p, load, circuit, cfg, initial_soc) for p in pop])

        for i, ei in enumerate(elite_idx):
            worst = np.argmax(new_fits)
            if fits[ei] < new_fits[worst]:
                pop[worst]      = pop[ei].copy()
                new_fits[worst] = fits[ei]

        fits = new_fits
        cur_best = np.argmin(fits)
        if fits[cur_best] < best_fit:
            best_fit = fits[cur_best]
            best_x   = pop[cur_best].copy()

    # Decode best solution
    diesel_on = np.round(best_x[:24].clip(0, 1))
    bess_mw   = best_x[24:].clip(-bc["charge_rate_mw"],
                                  (bc["soc_max"] - bc["soc_min"]) * cap)
    opt_mw = dc["optimal_output_mw"]
    rated  = dc["rated_mw"]
    curve  = {float(k): v for k, v in dc["bsfc_curve"].items()}

    schedule, soc = [], initial_soc
    total_fuel = total_carbon = 0.0
    for h in range(24):
        d_mw = diesel_on[h] * opt_mw
        b_mw = bess_mw[h]
        soc  = np.clip(soc - b_mw / bc["capacity_mwh"],
                       bc["soc_min"], bc["soc_max"])
        lf = d_mw / rated if d_mw > 0 else 0.0
        sfc = _bsfc(lf, curve) if lf > 0 else 0.0
        fuel_kg = sfc * d_mw / 1000.0
        co2_kg  = fuel_kg * 2.68
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
        "diesel_hours":   int(diesel_on.sum()),
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
