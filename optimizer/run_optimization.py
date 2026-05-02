"""
PEA Dispatch Optimization — Test Runner

Loads the hybrid forecast from data/test.parquet, runs the MILP optimizer
over the first 7 days (168 hours, 7 × 24h windows), and prints a full report.

Usage:
    python optimizer/run_optimization.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yaml

from optimizer.pea_dispatch_opt import pea_optimize, T, _bsfc_interp
from optimizer.dispatch import run_dispatch, schedule_summary   # rule-based baseline


def load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def get_forecast(cfg, n_days=7, use_kireip=False):
    """
    Load forecast data. use_kireip=True loads the KIREIP proxy dataset
    (real-world calibrated) instead of synthetic test.parquet.
    """
    if use_kireip and os.path.exists("data/kireip_proxy.parquet"):
        df = pd.read_parquet("data/kireip_proxy.parquet")
        n  = n_days * T
        load    = df["Island_Load_MW"].values[:n]
        circuit = df["Circuit_Cap_MW"].values[:n]
        print("  [Dataset] KIREIP proxy (real-world calibrated)")
    else:
        df = pd.read_parquet("data/test.parquet")
        n  = n_days * T
        load    = df["Island_Load_MW"].values[:n]
        cap_max = cfg["data"]["circuit_cap_max"]
        hours   = df.index[:n].hour if hasattr(df.index, "hour") else \
                  pd.to_datetime(df["Timestamp"].values[:n]).hour
        circuit = np.where(np.isin(hours, cfg["data"]["bottleneck_hours"]),
                           cap_max * 0.30, cap_max)
        print("  [Dataset] Synthetic test.parquet")
    return load, circuit


def reactive_baseline_cost(load, circuit, cfg):
    """
    PEA legacy baseline: diesel runs 24/7 as spinning reserve (always-on).
    This matches real pre-AI dispatch on Thai island grids where diesel is
    kept online continuously to guarantee supply — no BESS coordination.

    Dispatch rule:
      - Diesel always on at minimum 75% rated (spinning reserve floor)
      - Diesel ramps up to cover any load not met by circuit capacity
      - No BESS — circuit capacity is the only renewable proxy
    """
    dc    = cfg["diesel"]
    oc    = cfg["optimizer"]
    rated = dc["rated_mw"]
    opt   = dc["optimal_output_mw"]          # 7.5 MW — most efficient point
    curve = {float(k): v for k, v in dc["bsfc_curve"].items()}
    spin_floor = 0.75 * rated                # 7.5 MW spinning reserve floor

    total_fuel = total_carbon = 0.0
    diesel_hours = len(load)                 # always on

    for h in range(len(load)):
        # Renewable covers what it can; diesel covers the rest, min spin_floor
        renewable_avail = min(circuit[h], load[h])
        diesel_needed   = max(load[h] - renewable_avail, 0.0)
        d_mw = max(spin_floor, min(diesel_needed, rated))
        lf   = d_mw / rated
        sfc  = _bsfc_interp(lf, curve)
        fuel_kg = sfc * d_mw / 1000.0
        total_fuel   += fuel_kg
        total_carbon += fuel_kg * 2.68

    cost = total_fuel * oc["diesel_price_per_kg"] + \
           total_carbon * oc["carbon_price_per_kg"]
    return cost, total_fuel, total_carbon, diesel_hours


def print_day_table(day: int, result: dict):
    print(f"\n  Day {day+1} — cost {result['total_cost_thb']:.2f} THB | "
          f"fuel {result['total_fuel_kg']:.1f} kg | "
          f"CO₂ {result['total_carbon_kg']:.1f} kg | "
          f"diesel {result['diesel_hours']}h | "
          f"SoC_final {result['bess_soc_final']*100:.1f}%")
    print(f"  {'h':>3} {'load':>6} {'circ':>6} {'diesel':>7} {'on':>3} "
          f"{'bess':>7} {'SoC%':>6} {'fuel':>6} {'cost':>7}")
    print("  " + "─" * 62)
    for s in result["schedule"]:
        soc_pct = s.soc_mwh / 50 * 100   # cap=50 MWh
        print(f"  {s.hour:>3} {s.load_mw:>6.2f} {s.circuit_mw:>6.2f} "
              f"{s.diesel_mw:>7.2f} {s.diesel_on:>3} "
              f"{s.bess_mw:>+7.2f} {soc_pct:>6.1f} "
              f"{s.fuel_kg:>6.2f} {s.cost_thb:>7.2f}")


def main():
    cfg = load_cfg()
    n_days = 7
    use_kireip = os.path.exists("data/kireip_proxy.parquet")
    load_all, circuit_all = get_forecast(cfg, n_days, use_kireip=use_kireip)

    print("=" * 70)
    print("  PEA DISPATCH OPTIMIZATION — Ko Tao Microgrid")
    print(f"  Horizon: {n_days} days × 24h  |  Solver: HiGHS MILP")
    print("=" * 70)

    # ── Run MILP day-by-day (rolling 24h windows) ────────────────────────────
    milp_results = []
    soc = 0.65   # initial SoC fraction

    for day in range(n_days):
        load_d    = load_all[day*T:(day+1)*T]
        circuit_d = circuit_all[day*T:(day+1)*T]
        res = pea_optimize(load_d, circuit_d, initial_soc=soc, cfg=cfg)
        milp_results.append(res)
        soc = res["bess_soc_final"]   # carry SoC forward
        print_day_table(day, res)

    # ── Aggregate MILP totals ────────────────────────────────────────────────
    milp_cost    = sum(r["total_cost_thb"]  for r in milp_results)
    milp_fuel    = sum(r["total_fuel_kg"]   for r in milp_results)
    milp_carbon  = sum(r["total_carbon_kg"] for r in milp_results)
    milp_diesel_h = sum(r["diesel_hours"]   for r in milp_results)

    # ── Rule-based baseline ──────────────────────────────────────────────────
    rb_cost, rb_fuel, rb_carbon, rb_diesel_h = reactive_baseline_cost(
        load_all, circuit_all, cfg
    )

    # ── Comparison report ────────────────────────────────────────────────────
    fuel_saving_pct   = (rb_fuel   - milp_fuel)   / (rb_fuel   + 1e-9) * 100
    carbon_saving_pct = (rb_carbon - milp_carbon) / (rb_carbon + 1e-9) * 100
    cost_saving_pct   = (rb_cost   - milp_cost)   / (rb_cost   + 1e-9) * 100

    print("\n" + "=" * 70)
    print("  SUMMARY COMPARISON")
    print(f"  {'Metric':<30} {'MILP Opt':>12} {'Rule-Based':>12} {'Saving':>10}")
    print("  " + "─" * 66)
    print(f"  {'Total cost (THB)':<30} {milp_cost:>12.2f} {rb_cost:>12.2f} "
          f"{cost_saving_pct:>9.1f}%")
    print(f"  {'Fuel consumed (kg)':<30} {milp_fuel:>12.1f} {rb_fuel:>12.1f} "
          f"{fuel_saving_pct:>9.1f}%")
    print(f"  {'CO₂ emitted (kg)':<30} {milp_carbon:>12.1f} {rb_carbon:>12.1f} "
          f"{carbon_saving_pct:>9.1f}%")
    print(f"  {'Diesel-on hours':<30} {milp_diesel_h:>12} {rb_diesel_h:>12}")
    print("=" * 70)

    # PEA target checks
    pea_fuel_target = 0.22   # ≥22% fuel saving
    print("\n  PEA TARGET CHECKS")
    fuel_ok   = fuel_saving_pct   >= pea_fuel_target * 100
    carbon_ok = carbon_saving_pct >= 0
    print(f"  Fuel saving ≥22%  : {fuel_saving_pct:.1f}%  → {'PASS ✅' if fuel_ok else 'FAIL ❌'}")
    print(f"  Carbon reduction  : {carbon_saving_pct:.1f}%  → {'PASS ✅' if carbon_ok else 'FAIL ❌'}")

    # ── Save JSON report ─────────────────────────────────────────────────────
    report = {
        "milp": {
            "total_cost_thb":  round(milp_cost, 2),
            "total_fuel_kg":   round(milp_fuel, 2),
            "total_carbon_kg": round(milp_carbon, 2),
            "diesel_hours":    milp_diesel_h,
        },
        "rule_based": {
            "total_cost_thb":  round(rb_cost, 2),
            "total_fuel_kg":   round(rb_fuel, 2),
            "total_carbon_kg": round(rb_carbon, 2),
            "diesel_hours":    rb_diesel_h,
        },
        "savings": {
            "cost_pct":   round(cost_saving_pct, 2),
            "fuel_pct":   round(fuel_saving_pct, 2),
            "carbon_pct": round(carbon_saving_pct, 2),
        },
        "pea_targets_met": {
            "fuel_saving_22pct": bool(fuel_ok),
            "carbon_reduction":  bool(carbon_ok),
        },
    }
    os.makedirs("results", exist_ok=True)
    out = "results/pea_optimization_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
