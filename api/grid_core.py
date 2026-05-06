"""
Core Library: Physical Grid Component Models.
Encapsulates state and logic for BESS, Diesel Generators, and Transmission Lines.
"""
import numpy as np
from typing import Dict, List, Optional
from pydantic import BaseModel


class BESSState(BaseModel):
    soc_mwh: float
    soc_pct: float
    is_charging: bool
    power_mw: float
    temp_c: float = 25.0


class BESS:
    def __init__(self, config: dict):
        self.capacity_mwh = config.get("capacity_mwh", 0.0)
        self.soc_min = config.get("soc_min", 0.20)
        self.soc_max = config.get("soc_max", 0.80)
        self.charge_rate_mw = config.get("charge_rate_mw", 0.0)
        self.eff = config.get("efficiency", 0.92)
        
        # Initial state
        self.soc_mwh = self.capacity_mwh * 0.65
        self.current_power_mw = 0.0
        self.temp_c = 25.0

    def step(self, power_request_mw: float, dt_h: float = 0.25) -> float:
        """
        Apply power request. Positive = Discharge, Negative = Charge.
        Returns actual power delivered/absorbed.
        """
        if self.capacity_mwh <= 0:
            return 0.0

        # Constraints
        max_p = self.charge_rate_mw
        p = np.clip(power_request_mw, -max_p, max_p)

        if p > 0: # Discharging
            available = (self.soc_mwh - self.soc_min * self.capacity_mwh) / (dt_h / self.eff)
            p = min(p, available)
            self.soc_mwh -= (p / self.eff) * dt_h
        else: # Charging
            room = (self.soc_max * self.capacity_mwh - self.soc_mwh) / (dt_h * self.eff)
            p = max(p, -room)
            self.soc_mwh -= (p * self.eff) * dt_h
        
        self.current_power_mw = p
        return p

    def get_state(self) -> BESSState:
        return BESSState(
            soc_mwh=round(self.soc_mwh, 3),
            soc_pct=round((self.soc_mwh / self.capacity_mwh * 100) if self.capacity_mwh > 0 else 0, 2),
            is_charging=self.current_power_mw < 0,
            power_mw=round(self.current_power_mw, 3),
            temp_c=self.temp_c
        )


class MultiGenPlant:
    """Manages a plant of 5 identical 2MW diesel units."""
    def __init__(self, config: dict, num_units: int = 5, unit_rating_mw: float = 2.0):
        self.num_units = num_units
        self.unit_rating_mw = unit_rating_mw
        self.min_load_mw = config.get("min_load_mw", 0.5) # Min load per unit
        self.ramp_rate = config.get("ramp_rate_mw_per_h", 3.0)
        self.bsfc_curve = config.get("bsfc_curve", {})
        
        self.units_active = 0
        self.current_p_mw = 0.0
        self.total_fuel_kg = 0.0
        self.runtimes = np.zeros(num_units)

    def calculate_fuel(self, p_mw: float, units: int, dt_h: float) -> float:
        if units <= 0 or p_mw <= 0: return 0.0
        # Load factor per active unit
        load_factor = (p_mw / units) / self.unit_rating_mw
        keys = sorted([float(k) for k in self.bsfc_curve.keys()])
        vals = [self.bsfc_curve[str(k) if str(k) in self.bsfc_curve else k] for k in keys]
        bsfc = np.interp(load_factor, keys, vals)
        return (bsfc * p_mw * 1000 * dt_h) / 1e6

    def step(self, target_mw: float, dt_h: float = 0.25) -> dict:
        """Determines required units and delivers power."""
        if target_mw <= 0:
            self.units_active = 0
            self.current_p_mw = 0.0
            return {"p_mw": 0.0, "units": 0}

        # Stacking logic: How many 2MW units do we need?
        # We try to keep units between 40% and 90% load for efficiency
        required_units = int(np.ceil(target_mw / self.unit_rating_mw))
        required_units = min(required_units, self.num_units)
        
        self.units_active = required_units
        
        # Ramp constraint (simplified for aggregate plant)
        max_ramp = self.ramp_rate * dt_h * self.num_units
        diff = target_mw - self.current_p_mw
        self.current_p_mw += np.clip(diff, -max_ramp, max_ramp)
        
        # Final physical limits
        self.current_p_mw = np.clip(self.current_p_mw, 0, self.units_active * self.unit_rating_mw)
        
        fuel = self.calculate_fuel(self.current_p_mw, self.units_active, dt_h)
        self.total_fuel_kg += fuel
        self.runtimes[:self.units_active] += dt_h
        
        return {"p_mw": round(self.current_p_mw, 3), "units": self.units_active}


class TransmissionLine:
    def __init__(self, config: dict):
        self.v_kv = config.get("voltage_kv", 115.0)
        self.r_ohm_km = config.get("resistance_ohm_per_km", 0.05)
        self.length_km = config.get("length_km", 23.25)
        self.max_mw = config.get("max_mw", 160.0)

    def estimate_losses(self, p_mw: float) -> float:
        """Estimate I^2R losses in MW."""
        if p_mw <= 0: return 0.0
        # I = P / (sqrt(3) * V * pf) -> assume pf=0.95
        i_amps = (p_mw * 1e6) / (1.732 * self.v_kv * 1e3 * 0.95)
        r_total = self.r_ohm_km * self.length_km
        losses_w = 3 * (i_amps**2) * r_total
        return losses_w / 1e6

    def get_loading_pct(self, p_mw: float) -> float:
        return (p_mw / self.max_mw) * 100


class IslandGrid:
    """Manages full state of a single island microgrid."""
    def __init__(self, island_name: str, cfg: dict):
        self.name = island_name
        self.cfg = cfg
        assets = cfg["cluster"]["assets"].get(island_name.lower().replace(" ", "_"), {})
        
        # Merge global configs with island specific overrides
        bess_cfg = cfg["bess"].copy()
        bess_cfg.update({"capacity_mwh": assets.get("bess_mwh", 0)})
        self.bess = BESS(bess_cfg)
        
        # 5x2MW Multi-Gen Configuration
        diesel_cfg = cfg["diesel"].copy()
        self.diesel = MultiGenPlant(diesel_cfg, num_units=5, unit_rating_mw=2.0)
        
        line_cfg = cfg["spatial_fidelity"]["mainland_link"].copy()
        self.main_link = TransmissionLine(line_cfg)
        
        self.current_load_mw = 0.0
        self.circuit_limit_mw = cfg["data"]["circuit_cap_max"]

    def update(self, load_mw: float, circuit_limit_mw: float, dt_h: float = 0.25):
        self.current_load_mw = load_mw
        self.circuit_limit_mw = circuit_limit_mw
        
        # Optimized Dispatch Logic: Lowest Cost Priority
        net_demand = load_mw - circuit_limit_mw
        
        if net_demand > 0:
            # Cost of BESS
            cap_mwh = self.bess.capacity_mwh
            depth = self.bess.soc_max - self.bess.soc_min
            oc = self.cfg["optimizer"]
            deg_per_mwh = oc["bess_degradation_cost_per_cycle"] / (cap_mwh * depth) if cap_mwh > 0 else 9999.0
            cost_bess_per_mw = deg_per_mwh * dt_h
            
            # Cost of Diesel
            # Estimate running 1 unit at needed load (or min load)
            p_min_unit = self.diesel.min_load_mw
            test_diesel_mw = max(net_demand, p_min_unit)
            fuel_kg_per_h = self.diesel.calculate_fuel(test_diesel_mw, 1, 1.0) # fuel per hour
            cost_per_kg = oc["diesel_price_per_kg"] + 2.68 * oc["carbon_price_per_kg"]
            cost_diesel_per_mw = (fuel_kg_per_h * cost_per_kg * dt_h) / test_diesel_mw
            
            bess_available = (self.bess.soc_mwh > self.bess.soc_min * cap_mwh + 0.05) and (cap_mwh > 0)
            
            if cost_bess_per_mw <= cost_diesel_per_mw and bess_available:
                p_bess = self.bess.step(net_demand, dt_h)
                rem = net_demand - p_bess
                
                if rem > 0:
                    res = self.diesel.step(rem, dt_h)
                    gap = rem - res["p_mw"]
                    if gap > 0:
                        self.bess.step(p_bess + gap, dt_h)
            else:
                res = self.diesel.step(net_demand, dt_h)
                gap = net_demand - res["p_mw"]
                if gap > 0:
                    self.bess.step(gap, dt_h)
                elif res["p_mw"] > net_demand:
                    excess = res["p_mw"] - net_demand
                    self.bess.step(-excess, dt_h)
        else:
            # Surplus: Charge BESS
            self.bess.step(net_demand, dt_h)
            self.diesel.step(0, dt_h)

    def get_status(self) -> dict:
        return {
            "island": self.name,
            "load_mw": round(self.current_load_mw, 3),
            "circuit_limit_mw": round(self.circuit_limit_mw, 3),
            "bess": self.bess.get_state().dict(),
            "diesel": {
                "units_active": self.diesel.units_active,
                "power_mw": round(self.diesel.current_p_mw, 3),
                "total_fuel_kg": round(self.diesel.total_fuel_kg, 2),
                "unit_avg_load_pct": round((self.diesel.current_p_mw / (self.diesel.units_active * self.diesel.unit_rating_mw) * 100) if self.diesel.units_active > 0 else 0, 1)
            },
            "line_losses_mw": round(self.main_link.estimate_losses(min(self.current_load_mw, self.circuit_limit_mw)), 4)
        }
