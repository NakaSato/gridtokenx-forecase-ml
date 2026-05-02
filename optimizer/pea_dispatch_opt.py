"""
PEA Dispatch Optimization Model — Mixed-Integer Linear Program (MILP)

Physical model for Ko Tao islanded microgrid:
  - Circuit import from mainland (given, not controllable)
  - Diesel generator (dispatchable, binary on/off)
  - BESS (charge/discharge, SoC-bounded)
  - Curtailment slack: surplus that cannot be stored is spilled (free disposal)
  - Load-shedding slack: penalised heavily (should be zero in normal operation)

Decision variables (per hour h = 0..23):
  u[h]      ∈ {0,1}   diesel on/off
  p_d[h]    ≥ 0       diesel output MW
  p_bp[h]   ≥ 0       BESS discharge MW  (positive)
  p_bn[h]   ≥ 0       BESS charge MW     (positive magnitude, actual = −p_bn)
  s[h]      ≥ 0       SoC MWh
  spill[h]  ≥ 0       curtailed surplus MW
  shed[h]   ≥ 0       load-shedding MW (heavily penalised)

Power balance (equality):
  circuit[h] + p_d[h] + p_bp[h] − p_bn[h] − spill[h] − shed[h] = load[h]

SoC dynamics:
  s[h] = s[h-1] − p_bp[h] + p_bn[h]   (lossless, η absorbed into bounds)

Objective — minimise total operating cost:
  min  Σ_h [ fuel_cost(p_d[h]) + carbon_cost(p_d[h])
             + deg_per_mwh · p_bp[h]
             + spill_penalty · spill[h]
             + shed_penalty  · shed[h] ]

PEA constraints:
  C1  Power balance
  C2  Diesel bounds:   P_min·u[h] ≤ p_d[h] ≤ P_rated·u[h]
  C3  BESS discharge:  p_bp[h] ≤ P_bess_max
  C4  BESS charge:     p_bn[h] ≤ P_bess_max
  C5  SoC dynamics
  C6  SoC bounds:      SoC_min ≤ s[h] ≤ SoC_max
  C7  Diesel ramp:     |p_d[h] − p_d[h-1]| ≤ Ramp_MW
  C8  Min-up time      (linearised)
  C9  Min-down time    (linearised)
  C10 Daily cycle cap: Σ p_bp[h] ≤ 0.5·usable_cap

Solver: scipy.optimize.milp  (HiGHS backend, no extra install)
"""
from __future__ import annotations
import numpy as np
import yaml
from dataclasses import dataclass
from typing import List
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

# ── PEA operational parameters ───────────────────────────────────────────────
PEA_RAMP_MW    = 3.0    # MW/h diesel ramp limit (up and down)
PEA_MIN_UP_H   = 2      # minimum consecutive ON hours
PEA_MIN_DN_H   = 2      # minimum consecutive OFF hours
PEA_DIESEL_MIN = 2.0    # MW minimum stable output when ON
SPILL_PENALTY  = 0.001  # THB/MWh curtailment (near-zero, free disposal)
SHED_PENALTY   = 1e6    # THB/MWh load-shedding (very high)
T              = 24     # horizon hours


@dataclass
class HourResult:
    hour: int
    load_mw: float
    circuit_mw: float
    diesel_mw: float
    diesel_on: int
    bess_mw: float       # net: + discharge / − charge
    soc_mwh: float
    spill_mw: float
    shed_mw: float
    fuel_kg: float
    carbon_kg: float
    cost_thb: float


def _bsfc_interp(load_factor: float, curve: dict) -> float:
    keys = sorted(curve.keys())
    vals = [curve[k] for k in keys]
    return float(np.interp(max(load_factor, 0.0), keys, vals))


def _fuel_cost(p_d: float, rated: float, curve: dict,
               d_price: float, c_price: float) -> tuple[float, float, float]:
    """Return (fuel_kg, co2_kg, cost_thb)."""
    if p_d <= 0:
        return 0.0, 0.0, 0.0
    lf      = p_d / rated
    sfc     = _bsfc_interp(lf, curve)
    fuel_kg = sfc * p_d / 1000.0
    co2_kg  = fuel_kg * 2.68
    cost    = fuel_kg * d_price + co2_kg * c_price
    return fuel_kg, co2_kg, cost


def _build_milp(load: np.ndarray, circuit: np.ndarray,
                initial_soc_mwh: float, cfg: dict):
    """
    Variable layout (n = 7*T):
      [0:T]    u[h]      binary  diesel on/off
      [T:2T]   p_d[h]    cont    diesel MW ≥ 0
      [2T:3T]  p_bp[h]   cont    BESS discharge MW ≥ 0
      [3T:4T]  p_bn[h]   cont    BESS charge MW ≥ 0  (magnitude)
      [4T:5T]  s[h]      cont    SoC MWh
      [5T:6T]  spill[h]  cont    curtailment MW ≥ 0
      [6T:7T]  shed[h]   cont    load-shed MW ≥ 0
    """
    bc  = cfg["bess"]
    dc  = cfg["diesel"]
    oc  = cfg["optimizer"]

    cap         = bc["capacity_mwh"]
    soc_min_mwh = bc["soc_min"] * cap
    soc_max_mwh = bc["soc_max"] * cap
    rated       = dc["rated_mw"]
    d_price     = oc["diesel_price_per_kg"]
    c_price     = oc["carbon_price_per_kg"]
    deg_cost    = oc["bess_degradation_cost_per_cycle"]
    bess_max    = bc["charge_rate_mw"]
    curve       = {float(k): v for k, v in dc["bsfc_curve"].items()}

    n = 7 * T

    # ── Objective ────────────────────────────────────────────────────────────
    sfc_opt     = _bsfc_interp(0.75, curve)
    c_per_mw    = (sfc_opt / 1000.0) * (d_price + 2.68 * c_price)
    deg_per_mwh = deg_cost / (cap * (bc["soc_max"] - bc["soc_min"]))

    c_obj = np.zeros(n)
    c_obj[T:2*T]   = c_per_mw       # diesel fuel+carbon cost
    c_obj[2*T:3*T] = deg_per_mwh    # BESS discharge degradation
    # p_bn (charge) has zero cost — charging is free/beneficial
    c_obj[5*T:6*T] = SPILL_PENALTY  # curtailment
    c_obj[6*T:7*T] = SHED_PENALTY   # load-shedding

    # ── Integrality ──────────────────────────────────────────────────────────
    integrality = np.zeros(n)
    integrality[:T] = 1  # u[h] binary

    # ── Variable bounds ──────────────────────────────────────────────────────
    lb = np.zeros(n)
    ub = np.zeros(n)

    ub[:T]       = 1                          # u binary
    ub[T:2*T]    = rated                      # diesel MW
    ub[2*T:3*T]  = bess_max                   # BESS discharge
    ub[3*T:4*T]  = bess_max                   # BESS charge
    lb[4*T:5*T]  = soc_min_mwh
    ub[4*T:5*T]  = soc_max_mwh
    ub[5*T:6*T]  = float(max(circuit)) + rated + bess_max  # spill
    ub[6*T:7*T]  = float(max(load))                        # shed

    bounds = Bounds(lb=lb, ub=ub)

    # ── Constraints ──────────────────────────────────────────────────────────
    A_rows, lo_vals, hi_vals = [], [], []

    def add(row_dict: dict, lo: float, hi: float):
        r = np.zeros(n)
        for idx, val in row_dict.items():
            r[idx] += val
        A_rows.append(r)
        lo_vals.append(lo)
        hi_vals.append(hi)

    for h in range(T):
        u     = h
        pd    = T + h
        pbp   = 2*T + h   # discharge
        pbn   = 3*T + h   # charge (magnitude)
        s     = 4*T + h
        spill = 5*T + h
        shed  = 6*T + h

        # C1: power balance
        # circuit[h] + p_d + p_bp - p_bn - spill - shed = load[h]
        rhs = load[h] - circuit[h]
        add({pd: 1, pbp: 1, pbn: -1, spill: -1, shed: -1}, rhs, rhs)

        # C2: diesel bounds  P_min*u ≤ p_d ≤ P_rated*u
        add({pd: 1, u: -rated},         -np.inf, 0)   # p_d ≤ rated*u
        add({pd: 1, u: -PEA_DIESEL_MIN}, 0, np.inf)   # p_d ≥ P_min*u

        # C5: SoC dynamics  s[h] = s[h-1] - p_bp[h] + p_bn[h]
        if h == 0:
            # s[0] = s_init - p_bp[0] + p_bn[0]
            add({s: 1, pbp: 1, pbn: -1}, initial_soc_mwh, initial_soc_mwh)
        else:
            s_prev = 4*T + (h - 1)
            # s[h] - s[h-1] + p_bp[h] - p_bn[h] = 0
            add({s: 1, s_prev: -1, pbp: 1, pbn: -1}, 0, 0)

        # C7: diesel ramp
        if h > 0:
            pd_prev = T + (h - 1)
            add({pd: 1, pd_prev: -1}, -PEA_RAMP_MW, PEA_RAMP_MW)

    # C8/C9: min-up / min-down time
    for h in range(1, T):
        u_h   = h
        u_hm1 = h - 1
        for k in range(1, min(PEA_MIN_UP_H, T - h)):
            add({u_h: 1, u_hm1: -1, h + k: -1}, -np.inf, 0)
        for k in range(1, min(PEA_MIN_DN_H, T - h)):
            add({u_hm1: 1, u_h: -1, h + k: 1}, -np.inf, 1)

    # C10: daily BESS cycle cap
    daily_limit = 0.5 * (bc["soc_max"] - bc["soc_min"]) * cap
    row = np.zeros(n)
    row[2*T:3*T] = 1   # sum of discharge
    A_rows.append(row); lo_vals.append(-np.inf); hi_vals.append(daily_limit)

    A = csc_matrix(np.array(A_rows))
    constraints = LinearConstraint(A, lo_vals, hi_vals)

    meta = dict(rated=rated, curve=curve, d_price=d_price,
                c_price=c_price, cap=cap)
    return c_obj, integrality, bounds, constraints, meta


def pea_optimize(
    load: np.ndarray,
    circuit: np.ndarray,
    initial_soc: float = 0.65,
    cfg: dict | None = None,
) -> dict:
    """
    Run PEA MILP dispatch optimisation for a 24-hour horizon.

    Parameters
    ----------
    load        : 24-element forecast load array (MW)
    circuit     : 24-element available circuit capacity array (MW)
    initial_soc : BESS state-of-charge fraction (0–1)
    cfg         : parsed config.yaml dict (loaded from file if None)

    Returns
    -------
    dict with schedule (List[HourResult]), totals, and solver status
    """
    if cfg is None:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

    cap             = cfg["bess"]["capacity_mwh"]
    initial_soc_mwh = initial_soc * cap

    c_obj, integrality, bounds, constraints, meta = _build_milp(
        load, circuit, initial_soc_mwh, cfg
    )

    result = milp(
        c_obj,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"disp": False, "time_limit": 30.0},
    )

    if result.status not in (0, 1):
        raise RuntimeError(f"MILP solver failed: {result.message}")

    x      = result.x
    u_sol  = np.round(x[:T]).astype(int)
    pd_sol = x[T:2*T]
    pbp_sol = x[2*T:3*T]   # discharge
    pbn_sol = x[3*T:4*T]   # charge magnitude
    s_sol  = x[4*T:5*T]
    sp_sol = x[5*T:6*T]
    sh_sol = x[6*T:7*T]

    schedule: List[HourResult] = []
    total_cost = total_fuel = total_carbon = 0.0

    for h in range(T):
        fuel_kg, co2_kg, cost = _fuel_cost(
            pd_sol[h], meta["rated"], meta["curve"],
            meta["d_price"], meta["c_price"]
        )
        total_fuel   += fuel_kg
        total_carbon += co2_kg
        total_cost   += cost

        bess_net = pbp_sol[h] - pbn_sol[h]  # + discharge / − charge

        schedule.append(HourResult(
            hour=h,
            load_mw=round(float(load[h]), 3),
            circuit_mw=round(float(circuit[h]), 3),
            diesel_mw=round(float(pd_sol[h]), 3),
            diesel_on=int(u_sol[h]),
            bess_mw=round(float(bess_net), 3),
            soc_mwh=round(float(s_sol[h]), 3),
            spill_mw=round(float(sp_sol[h]), 3),
            shed_mw=round(float(sh_sol[h]), 3),
            fuel_kg=round(fuel_kg, 3),
            carbon_kg=round(co2_kg, 3),
            cost_thb=round(cost, 2),
        ))

    return {
        "schedule":        schedule,
        "total_cost_thb":  round(total_cost, 2),
        "total_fuel_kg":   round(total_fuel, 2),
        "total_carbon_kg": round(total_carbon, 2),
        "diesel_hours":    int(u_sol.sum()),
        "bess_soc_final":  round(float(s_sol[-1]) / cap, 4),
        "total_spill_mwh": round(float(sp_sol.sum()), 3),
        "total_shed_mwh":  round(float(sh_sol.sum()), 3),
        "solver_status":   result.message,
    }
