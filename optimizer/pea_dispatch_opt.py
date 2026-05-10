"""
GridTokenX Coordinated Cluster MILP Dispatch
==============================================
Optimizes a 3-island radial-serial chain: Khanom -> Samui -> Phangan -> Tao.
Aligned to Project Schema and time-varying cable capacities.

Topology & Constraints:
  1. Mainland (Khanom) -> Samui: P_main <= kmb_flow_mw (predicted headroom/limit).
  2. Samui -> Phangan: P_SP <= cable_flow_mw (predicted limit).
  3. Phangan -> Tao: P_PT <= capacity_mw (predicted thermal limit).
  4. Ko Tao: 10 MW Diesel Plant (tao_load_mw).
  5. Ko Phangan: 15 MW Mobile Generators (phangan_load_mw).
  6. Ko Samui: 25 MW Mobile Generators + 50 MWh BESS (samui_load_mw).
"""
from __future__ import annotations
import numpy as np
import yaml
from dataclasses import dataclass
from typing import List, Dict
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

# ── Constants ────────────────────────────────────────────────────────────────
SPILL_PENALTY = 0.001
SHED_PENALTY  = 1e6

@dataclass
class ClusterStepResult:
    hour: int
    tao_load: float
    pha_load: float
    sam_load: float
    tao_gen: float
    pha_gen: float
    sam_gen: float
    bess_mw: float
    bess_soc: float
    flow_main: float
    flow_sp: float
    flow_pt: float
    shed_total: float
    fuel_kg: float
    carbon_kg: float
    cost_thb: float

def _bsfc_interp(load_factor: float, curve: dict) -> float:
    keys = sorted(curve.keys())
    vals = [curve[k] for k in keys]
    return float(np.interp(max(load_factor, 0.0), keys, vals))

def _fit_linear_fuel(rated: float, curve: dict) -> tuple[float, float]:
    lf1, lf2 = 0.25, 0.75
    p1, p2 = lf1 * rated, lf2 * rated
    f1 = _bsfc_interp(lf1, curve) * p1
    f2 = _bsfc_interp(lf2, curve) * p2
    slope = (f2 - f1) / (p2 - p1)
    intercept = f2 - slope * p2
    return max(0.0, intercept), slope

def _fuel_cost(p_d: float, rated: float, curve: dict, d_price: float, c_price: float):
    if p_d <= 0.001: return 0.0, 0.0, 0.0
    lf = p_d / rated
    sfc = _bsfc_interp(lf, curve)
    fuel_kg = sfc * p_d
    co2_kg = fuel_kg * 2.68
    cost = fuel_kg * (d_price + 2.68 * c_price)
    return fuel_kg, co2_kg, cost

def _build_cluster_milp(
    loads: Dict[str, np.ndarray], 
    caps: Dict[str, np.ndarray],
    initial_soc_mwh: float, 
    cfg: dict
):
    T = len(loads["tao"])
    n = 14 * T
    
    bc = cfg["bess"]
    dc = cfg["diesel"]
    oc = cfg["optimizer"]
    cl = cfg["cluster"]["assets"]
    freq = cfg["data"].get("frequency", "h")
    sph = 4 if freq == "15min" else 1
    dt = 1.0 / sph

    tao_rated = cl["ko_tao"]["diesel_mw"]
    pha_rated = cl["ko_phangan"]["diesel_mw"]
    sam_rated = cl["ko_samui"]["diesel_mw"]
    bess_cap  = bc["capacity_mwh"]
    bess_max  = bc["charge_rate_mw"]
    
    curve = {float(k): v for k, v in dc["bsfc_curve"].items()}
    f_fix, f_slope = _fit_linear_fuel(tao_rated, curve)
    cost_per_kg = oc["diesel_price_per_kg"] + 2.68 * oc["carbon_price_per_kg"]
    
    c_obj = np.zeros(n)
    # Fuel costs (Tao, Phangan, Samui)
    c_obj[0:T]      = (f_fix * cost_per_kg) * dt
    c_obj[T:2*T]    = (f_slope * cost_per_kg) * dt
    c_obj[2*T:3*T]  = (f_fix * cost_per_kg * 1.1) * dt # Slight premium for mobile gens
    c_obj[3*T:4*T]  = (f_slope * cost_per_kg * 1.1) * dt
    c_obj[4*T:5*T]  = (f_fix * cost_per_kg * 1.2) * dt
    c_obj[5*T:6*T]  = (f_slope * cost_per_kg * 1.2) * dt
    
    depth = bc["soc_max"] - bc["soc_min"]
    deg_per_mwh = oc["bess_degradation_cost_per_cycle"] / (bess_cap * depth + 1e-6)
    c_obj[6*T:7*T] = deg_per_mwh * dt # Charge
    c_obj[7*T:8*T] = deg_per_mwh * dt # Discharge
    
    c_obj[12*T:13*T] = SHED_PENALTY * dt
    c_obj[13*T:14*T] = SPILL_PENALTY * dt

    integrality = np.zeros(n)
    integrality[0:T] = 1; integrality[2*T:3*T] = 1; integrality[4*T:5*T] = 1

    lb, ub = np.zeros(n), np.full(n, np.inf)
    ub[0:T] = 1; ub[T:2*T] = tao_rated
    ub[2*T:3*T] = 1; ub[3*T:4*T] = pha_rated
    ub[4*T:5*T] = 1; ub[5*T:6*T] = sam_rated
    ub[6*T:7*T] = bess_max; ub[7*T:8*T] = bess_max
    lb[8*T:9*T] = bc["soc_min"] * bess_cap; ub[8*T:9*T] = bc["soc_max"] * bess_cap
    
    # ── Inter-island Flow Limits (Time-varying) ──
    ub[9*T:10*T] = caps["kmb"]      # P_main
    ub[10*T:11*T] = caps["cable"]   # P_SP
    ub[11*T:12*T] = caps["capacity"] # P_PT
    
    bounds = Bounds(lb, ub)

    A_rows, lo_vals, hi_vals = [], [], []
    def add(row_dict: dict, lo: float, hi: float):
        r = np.zeros(n)
        for idx, val in row_dict.items(): r[idx] += val
        A_rows.append(r); lo_vals.append(lo); hi_vals.append(hi)

    for h in range(T):
        ut, pt = h, T + h
        up, pp = 2*T + h, 3*T + h
        us, ps = 4*T + h, 5*T + h
        pbp, pbn, s = 6*T + h, 7*T + h, 8*T + h
        P_main, P_SP, P_PT = 9*T + h, 10*T + h, 11*T + h
        shed, spill = 12*T + h, 13*T + h

        # Balance Equations
        add({P_PT: 1, pt: 1, shed: 1, spill: -1}, loads["tao"][h], loads["tao"][h])
        add({P_SP: 1, P_PT: -1, pp: 1}, loads["pha"][h], loads["pha"][h])
        add({P_main: 1, P_SP: -1, ps: 1, pbp: 1, pbn: -1}, loads["sam"][h], loads["sam"][h])

        # Commitment/Capacity Constraints
        p_min = dc.get("min_load_mw", 2.0)
        add({pt: 1, ut: -tao_rated}, -np.inf, 0); add({pt: 1, ut: -p_min}, 0, np.inf)
        add({pp: 1, up: -pha_rated}, -np.inf, 0); add({pp: 1, up: -p_min}, 0, np.inf)
        add({ps: 1, us: -sam_rated}, -np.inf, 0); add({ps: 1, us: -p_min}, 0, np.inf)

        # BESS SoC Dynamics
        if h == 0:
            add({s: 1, pbp: dt, pbn: -dt}, initial_soc_mwh, initial_soc_mwh)
        else:
            add({s: 1, (8*T + h - 1): -1, pbp: dt, pbn: -dt}, 0, 0)

    A = csc_matrix(np.array(A_rows))
    constraints = LinearConstraint(A, lo_vals, hi_vals)
    return c_obj, integrality, bounds, constraints, {
        "tao_rated": tao_rated, "pha_rated": pha_rated, "sam_rated": sam_rated,
        "curve": curve, "d_price": oc["diesel_price_per_kg"], "c_price": oc["carbon_price_per_kg"],
        "bess_cap": bess_cap, "sph": sph
    }

def cluster_optimize(loads: Dict[str, np.ndarray], kmb_headroom: np.ndarray,
                     initial_soc: float = 0.65, cfg: dict | None = None,
                     cable_cap: np.ndarray | None = None, 
                     distal_cap: np.ndarray | None = None) -> dict:
    if cfg is None:
        with open("config.yaml") as f: cfg = yaml.safe_load(f)
    
    T = len(loads["tao"])
    caps = {
        "kmb": kmb_headroom,
        "cable": cable_cap if cable_cap is not None else np.full(T, 30.0),
        "capacity": distal_cap if distal_cap is not None else np.full(T, 16.0)
    }
    
    bess_cap = cfg["bess"]["capacity_mwh"]
    c_obj, integrality, bounds, constraints, meta = _build_cluster_milp(
        loads, caps, initial_soc * bess_cap, cfg
    )
    result = milp(c_obj, constraints=constraints, integrality=integrality, bounds=bounds,
                  options={"time_limit": 60.0})
    if not result.success:
        return {"status": "FAILED", "message": result.message}
    
    x = result.x
    pt_sol = x[T:2*T]; pp_sol = x[3*T:4*T]; ps_sol = x[5*T:6*T]
    pbp_sol = x[6*T:7*T]; pbn_sol = x[7*T:8*T]; s_sol = x[8*T:9*T]
    P_main_sol = x[9*T:10*T]; P_SP_sol = x[10*T:11*T]; P_PT_sol = x[11*T:12*T]
    shed_sol = x[12*T:13*T]
    
    schedule = []
    total_fuel = total_carbon = 0.0
    sph = meta["sph"]; dt = 1.0 / sph
    for h in range(T):
        f_t, c_t, _ = _fuel_cost(pt_sol[h], meta["tao_rated"], meta["curve"], 0, 0)
        f_p, c_p, _ = _fuel_cost(pp_sol[h], meta["pha_rated"], meta["curve"], 0, 0)
        f_s, c_s, _ = _fuel_cost(ps_sol[h], meta["sam_rated"], meta["curve"], 0, 0)
        step_fuel = (f_t + f_p + f_s) * dt; step_carb = (c_t + c_p + c_s) * dt
        total_fuel += step_fuel; total_carbon += step_carb
        schedule.append(ClusterStepResult(
            hour=h, tao_load=loads["tao"][h], pha_load=loads["pha"][h], sam_load=loads["sam"][h],
            tao_gen=pt_sol[h], pha_gen=pp_sol[h], sam_gen=ps_sol[h],
            bess_mw=pbp_sol[h] - pbn_sol[h], bess_soc=s_sol[h] / (bess_cap + 1e-6) if bess_cap > 0 else 0.0,
            flow_main=P_main_sol[h], flow_sp=P_SP_sol[h], flow_pt=P_PT_sol[h],
            shed_total=shed_sol[h], fuel_kg=step_fuel, carbon_kg=step_carb,
            cost_thb=step_fuel * (meta["d_price"] + 2.68 * meta["c_price"])
        ))
    return {"status": "SUCCESS", "schedule": schedule, "total_fuel_kg": round(total_fuel, 2),
            "total_carbon_kg": round(total_carbon, 2), "total_cost_thb": round(sum(s.cost_thb for s in schedule), 2),
            "bess_soc_final": schedule[-1].bess_soc if bess_cap > 0 else initial_soc}

def pea_optimize(load, circuit, initial_soc=0.65, cfg=None):
    loads = {"tao": load, "pha": np.zeros_like(load), "sam": np.zeros_like(load)}
    # Backward compatibility for single-island PoC
    res = cluster_optimize(loads, kmb_headroom=np.full_like(load, 174.0), 
                           initial_soc=initial_soc, cfg=cfg, distal_cap=circuit)
    if res["status"] == "FAILED": return {"solver_status": res["message"]}
    return res
