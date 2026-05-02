"""
Tests for optimizer/early_warning.py — Early Warning System.
"""
import os
import sys
import pytest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from optimizer.early_warning import check_warnings, format_warnings, Warning


class TestCheckWarnings:
    """Tests for the early warning detection logic."""

    def test_no_warnings_stable_grid(self, cfg):
        """No warnings when load is well below circuit capacity."""
        load = np.full(24, 6.0)       # low load
        circuit = np.full(24, 14.0)   # high capacity
        warnings = check_warnings(load, circuit, current_soc=0.65, cfg=cfg)
        # Should have zero CRITICAL/WARNING (may have INFO about no bottleneck)
        critical = [w for w in warnings if w.level in ("CRITICAL", "WARNING")]
        assert len(critical) == 0

    def test_critical_bess_depletion(self, cfg):
        """CRITICAL warning when load >> circuit and SoC at minimum."""
        load = np.full(24, 10.0)      # high load
        circuit = np.full(24, 2.0)    # severe bottleneck
        # Start with very low SoC
        warnings = check_warnings(load, circuit, current_soc=0.21, cfg=cfg,
                                   lookahead_hours=6)
        critical = [w for w in warnings if w.level == "CRITICAL"]
        assert len(critical) > 0, "No CRITICAL warning for BESS depletion scenario"

    def test_warning_low_soc(self, cfg):
        """WARNING when SoC approaches minimum."""
        load = np.full(24, 8.0)
        circuit = np.full(24, 5.0)    # moderate deficit
        warnings = check_warnings(load, circuit, current_soc=0.35, cfg=cfg,
                                   lookahead_hours=6)
        # Should trigger at least a WARNING
        warn_or_crit = [w for w in warnings if w.level in ("CRITICAL", "WARNING")]
        assert len(warn_or_crit) > 0

    def test_info_bottleneck_detected(self, cfg):
        """INFO warning when circuit drops below 5 MW in forecast window."""
        load = np.full(24, 7.0)
        circuit = np.full(24, 14.0)
        circuit[18:22] = 3.0  # bottleneck window
        warnings = check_warnings(load, circuit, current_soc=0.65, cfg=cfg)
        info = [w for w in warnings if w.level == "INFO"]
        assert len(info) > 0, "No INFO warning for detected bottleneck"
        assert "Bottleneck" in info[0].message or "bottleneck" in info[0].message.lower()

    def test_warning_hour_correct(self, cfg):
        """Warning hour should correspond to when the event occurs."""
        load = np.full(24, 10.0)
        circuit = np.full(24, 2.0)
        warnings = check_warnings(load, circuit, current_soc=0.25, cfg=cfg,
                                   lookahead_hours=6)
        for w in warnings:
            if w.level in ("CRITICAL", "WARNING"):
                assert 0 <= w.hour < 6, f"Warning hour {w.hour} outside lookahead"

    def test_soc_at_event_within_bounds(self, cfg):
        """SoC recorded in warnings should be within physical bounds."""
        load = np.full(24, 10.0)
        circuit = np.full(24, 3.0)
        warnings = check_warnings(load, circuit, current_soc=0.40, cfg=cfg,
                                   lookahead_hours=6)
        bc = cfg["bess"]
        for w in warnings:
            assert w.soc_at_event >= bc["soc_min"] - 0.001
            assert w.soc_at_event <= bc["soc_max"] + 0.001

    def test_lookahead_respected(self, cfg):
        """Only check within the lookahead window."""
        load = np.full(24, 10.0)
        circuit = np.full(24, 2.0)
        w3 = check_warnings(load, circuit, current_soc=0.65, cfg=cfg,
                             lookahead_hours=3)
        w12 = check_warnings(load, circuit, current_soc=0.65, cfg=cfg,
                              lookahead_hours=12)
        # Wider lookahead should capture at least as many warnings
        crit_3 = [w for w in w3 if w.level != "INFO"]
        crit_12 = [w for w in w12 if w.level != "INFO"]
        assert len(crit_12) >= len(crit_3)


class TestFormatWarnings:
    """Tests for warning message formatting."""

    def test_no_warnings_message(self):
        result = format_warnings([])
        assert "No warnings" in result
        assert "✅" in result

    def test_critical_first(self):
        """CRITICAL warnings should appear before WARNING and INFO."""
        warnings = [
            Warning("INFO", 5, "Info message", 0.5),
            Warning("CRITICAL", 1, "Critical message", 0.2),
            Warning("WARNING", 3, "Warning message", 0.35),
        ]
        result = format_warnings(warnings)
        lines = result.strip().split("\n")
        assert "CRITICAL" in lines[0]

    def test_icons_present(self):
        warnings = [
            Warning("CRITICAL", 0, "Test", 0.2),
            Warning("WARNING", 1, "Test", 0.3),
            Warning("INFO", 2, "Test", 0.5),
        ]
        result = format_warnings(warnings)
        assert "🔴" in result
        assert "🟡" in result
        assert "🔵" in result
