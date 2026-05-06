"""
Tests for optimizer/dispatch.py — rule-based predictive dispatch logic.
"""
import os
import sys
import pytest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from optimizer.dispatch import run_dispatch, schedule_summary, _bsfc, HourlyDispatch


class TestBSFC:
    """Tests for the BSFC interpolation function."""

    def test_known_points(self, cfg):
        curve = {float(k): v for k, v in cfg["diesel"]["bsfc_curve"].items()}
        # At 75% load factor, BSFC should be ~198.5 g/kWh
        assert abs(_bsfc(0.75, curve) - 198.5) < 0.1

    def test_interpolation_between_points(self, cfg):
        curve = {float(k): v for k, v in cfg["diesel"]["bsfc_curve"].items()}
        # Between 50% (225) and 75% (198.5), interpolated value
        val = _bsfc(0.625, curve)
        assert 198.5 < val < 225.0

    def test_zero_load_factor(self, cfg):
        curve = {float(k): v for k, v in cfg["diesel"]["bsfc_curve"].items()}
        # At 0% — extrapolated/boundary
        val = _bsfc(0.0, curve)
        assert val >= 0

    def test_full_load(self, cfg):
        curve = {float(k): v for k, v in cfg["diesel"]["bsfc_curve"].items()}
        assert abs(_bsfc(1.0, curve) - 210.0) < 0.1


class TestRunDispatch:
    """Tests for the main dispatch scheduling logic."""

    def test_output_length(self, sample_load_24h, sample_circuit_24h, cfg):
        schedule = run_dispatch(sample_load_24h, sample_circuit_24h, cfg=cfg)
        assert len(schedule) == 24

    def test_output_types(self, sample_load_24h, sample_circuit_24h, cfg):
        schedule = run_dispatch(sample_load_24h, sample_circuit_24h, cfg=cfg)
        for s in schedule:
            assert isinstance(s, HourlyDispatch)

    def test_soc_within_bounds(self, sample_load_24h, sample_circuit_24h, cfg):
        """SoC must always stay within [soc_min, soc_max]."""
        schedule = run_dispatch(sample_load_24h, sample_circuit_24h, cfg=cfg)
        bc = cfg["bess"]
        for s in schedule:
            assert s.bess_soc >= bc["soc_min"] - 1e-6, \
                f"SoC {s.bess_soc} below min {bc['soc_min']} at h{s.hour}"
            assert s.bess_soc <= bc["soc_max"] + 1e-6, \
                f"SoC {s.bess_soc} above max {bc['soc_max']} at h{s.hour}"

    def test_no_diesel_when_surplus(self, sample_load_24h, sample_circuit_no_bottleneck, cfg):
        """When circuit >> load, diesel should not fire."""
        schedule = run_dispatch(sample_load_24h, sample_circuit_no_bottleneck, cfg=cfg)
        for s in schedule:
            assert s.diesel_mw == 0.0, f"Diesel fired at h{s.hour} despite surplus"
            assert s.fuel_kg == 0.0

    def test_bess_charges_during_surplus(self, sample_load_24h, sample_circuit_no_bottleneck, cfg):
        """During surplus (circuit > load), BESS should charge (negative bess_mw)."""
        if cfg["bess"]["capacity_mwh"] <= 0:
            pytest.skip("BESS capacity is zero — skip charging test")
        
        schedule = run_dispatch(sample_load_24h, sample_circuit_no_bottleneck,
                                initial_soc=0.30, cfg=cfg)
        # At least some hours should show charging
        charging_hours = [s for s in schedule if s.bess_mw < 0]
        assert len(charging_hours) > 0, "BESS never charges despite surplus"

    def test_diesel_fires_under_large_deficit(self, high_deficit_load, low_circuit, cfg):
        """When load greatly exceeds circuit + BESS capacity, diesel must fire."""
        schedule = run_dispatch(high_deficit_load, low_circuit,
                                initial_soc=0.30, cfg=cfg)
        diesel_hours = [s for s in schedule if s.diesel_mw > 0]
        assert len(diesel_hours) > 0, "Diesel never fires despite large deficit"

    def test_diesel_at_optimal_output(self, high_deficit_load, low_circuit, cfg):
        """When diesel fires, it should prioritize optimal_output_mw (7.5 MW) if BESS available."""
        schedule = run_dispatch(high_deficit_load, low_circuit, cfg=cfg)
        opt_mw = cfg["diesel"]["optimal_output_mw"]
        has_bess = cfg["bess"]["capacity_mwh"] > 0
        
        for s in schedule:
            if s.diesel_mw > 0:
                if has_bess:
                    # With BESS, diesel should stay near optimal
                    assert s.diesel_mw <= opt_mw + 0.001, \
                        f"Diesel {s.diesel_mw} exceeds optimal {opt_mw} with BESS available"
                else:
                    # Without BESS, diesel must cover the full deficit up to rated
                    delta = s.load_mw - s.circuit_mw
                    assert s.diesel_mw >= min(delta, cfg["diesel"]["rated_mw"]) - 0.001

    def test_fuel_positive_when_diesel_on(self, high_deficit_load, low_circuit, cfg):
        """Fuel consumption must be > 0 whenever diesel is running."""
        schedule = run_dispatch(high_deficit_load, low_circuit, cfg=cfg)
        for s in schedule:
            if s.diesel_mw > 0:
                assert s.fuel_kg > 0, f"Zero fuel at h{s.hour} with diesel={s.diesel_mw}"
                assert s.carbon_kg > 0, f"Zero carbon at h{s.hour}"

    def test_fuel_zero_when_diesel_off(self, sample_load_24h, sample_circuit_no_bottleneck, cfg):
        """No fuel or carbon when diesel is off."""
        schedule = run_dispatch(sample_load_24h, sample_circuit_no_bottleneck, cfg=cfg)
        for s in schedule:
            if s.diesel_mw == 0:
                assert s.fuel_kg == 0.0
                assert s.carbon_kg == 0.0

    def test_initial_soc_respected(self, sample_load_24h, sample_circuit_24h, cfg):
        """Different initial SoC should produce different schedules."""
        sched_low = run_dispatch(sample_load_24h, sample_circuit_24h,
                                  initial_soc=0.25, cfg=cfg)
        sched_high = run_dispatch(sample_load_24h, sample_circuit_24h,
                                   initial_soc=0.75, cfg=cfg)
        # First hour SoC should differ
        assert sched_low[0].bess_soc != sched_high[0].bess_soc

    def test_carbon_proportional_to_fuel(self, high_deficit_load, low_circuit, cfg):
        """CO2 should be fuel_kg * 2.68 (diesel emission factor)."""
        schedule = run_dispatch(high_deficit_load, low_circuit, cfg=cfg)
        for s in schedule:
            if s.fuel_kg > 0:
                expected_co2 = round(s.fuel_kg * 2.68, 3)
                assert abs(s.carbon_kg - expected_co2) < 0.01, \
                    f"h{s.hour}: carbon {s.carbon_kg} != {expected_co2}"


class TestScheduleSummary:
    """Tests for schedule_summary aggregation."""

    def test_summary_keys(self, sample_load_24h, sample_circuit_24h, cfg):
        schedule = run_dispatch(sample_load_24h, sample_circuit_24h, cfg=cfg)
        summary = schedule_summary(schedule)
        expected_keys = {"total_fuel_kg", "total_carbon_kg", "diesel_hours", "bess_soc_final"}
        assert set(summary.keys()) == expected_keys

    def test_fuel_sum_matches(self, high_deficit_load, low_circuit, cfg):
        schedule = run_dispatch(high_deficit_load, low_circuit, cfg=cfg)
        summary = schedule_summary(schedule)
        manual_fuel = sum(s.fuel_kg for s in schedule)
        assert abs(summary["total_fuel_kg"] - manual_fuel) < 0.01

    def test_diesel_hours_count(self, high_deficit_load, low_circuit, cfg):
        schedule = run_dispatch(high_deficit_load, low_circuit, cfg=cfg)
        summary = schedule_summary(schedule)
        manual_count = sum(1 for s in schedule if s.diesel_mw > 0)
        assert summary["diesel_hours"] == manual_count

    def test_final_soc_matches(self, sample_load_24h, sample_circuit_24h, cfg):
        schedule = run_dispatch(sample_load_24h, sample_circuit_24h, cfg=cfg)
        summary = schedule_summary(schedule)
        assert summary["bess_soc_final"] == schedule[-1].bess_soc
