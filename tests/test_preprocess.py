"""
Tests for data/preprocess.py — feature engineering and splitting logic.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from data.preprocess import engineer_features, split, impute_bess_soc


def _make_sample_df(n_hours=500):
    """Create a minimal synthetic DataFrame matching island_grid.parquet schema."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01", periods=n_hours, freq="h")
    df = pd.DataFrame({
        "Island_Load_MW": rng.uniform(5, 10, n_hours),
        "Circuit_Cap_MW": rng.uniform(8, 16, n_hours),
        "Dry_Bulb_Temp": rng.uniform(24, 36, n_hours),
        "Rel_Humidity": rng.uniform(60, 95, n_hours),
        "Solar_Irradiance": rng.uniform(0, 1050, n_hours),
        "Wind_Speed": rng.uniform(0, 15, n_hours),
        "Cloud_Cover": rng.uniform(0, 100, n_hours),
        "Carbon_Intensity": rng.uniform(400, 850, n_hours),
        "Market_Price": rng.uniform(35, 120, n_hours),
        "Tourist_Index": rng.uniform(0.2, 1.0, n_hours),
        "BESS_SoC_Pct": rng.uniform(20, 95, n_hours),
        "Net_Delta_MW": rng.uniform(-5, 5, n_hours),
    }, index=idx)
    return df


@pytest.fixture
def sample_df():
    return _make_sample_df()


@pytest.fixture
def cfg_data():
    import yaml
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


class TestEngineerFeatures:
    """Tests for the feature engineering pipeline."""

    def test_lag_features_created(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        # Check for new schema names
        for lag in [1, 24]:
            assert f"tao_load_mw_lag_{lag}h" in result.columns, f"Missing tao_load_mw_lag_{lag}h"

    def test_rolling_features_created(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        for w in [3, 6]:
            assert f"tao_load_roll_mean_{w}h" in result.columns
            assert f"tao_load_roll_std_{w}h" in result.columns

    def test_heat_index_correct(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        assert "heat_index" in result.columns
        # Heat index = temp * humidity / 100
        expected = result["t2m_celsius"] * result["rh_pct"] / 100
        np.testing.assert_allclose(result["heat_index"].values, expected.values, rtol=1e-5)

    def test_time_features_created(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        assert "hour_of_day" in result.columns
        assert "day_of_week" in result.columns
        assert "is_high_season" in result.columns

    def test_hour_of_day_range(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        assert result["hour_of_day"].min() >= 0
        assert result["hour_of_day"].max() <= 23

    def test_is_high_season_binary(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        assert set(result["is_high_season"].unique()).issubset({0, 1})

    def test_leaky_features_dropped(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        # Net_Delta_MW should be gone (not in rename map and not numeric target)
        assert "Net_Delta_MW" not in result.columns
        # Circuit_Cap_MW renamed to capacity_mw
        assert "Circuit_Cap_MW" not in result.columns
        assert "capacity_mw" in result.columns

    def test_no_nan_after_dropna(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        assert result.isna().sum().sum() == 0, "NaN values remain after dropna"

    def test_weather_trends_created(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        assert "temp_gradient" in result.columns

    def test_output_rows_less_than_input(self, sample_df, cfg_data):
        """Feature engineering drops rows due to lags and rolling windows."""
        result = engineer_features(sample_df, cfg_data)
        assert len(result) < len(sample_df)


class TestSplit:
    """Tests for the chronological train/val/test split."""

    def test_split_ratios(self, sample_df, cfg_data):
        fe = engineer_features(sample_df, cfg_data)
        train, val, test = split(fe)
        total = len(fe)
        assert abs(len(train) / total - 0.70) < 0.05
        assert abs(len(val) / total - 0.15) < 0.05
        assert abs(len(test) / total - 0.15) < 0.05

    def test_no_overlap(self, sample_df, cfg_data):
        fe = engineer_features(sample_df, cfg_data)
        train, val, test = split(fe)
        assert train.index.max() < val.index.min(), "Train overlaps with val"
        assert val.index.max() < test.index.min(), "Val overlaps with test"

    def test_chronological_order(self, sample_df, cfg_data):
        fe = engineer_features(sample_df, cfg_data)
        train, val, test = split(fe)
        # Each split's index should be monotonically increasing
        assert train.index.is_monotonic_increasing
        assert val.index.is_monotonic_increasing
        assert test.index.is_monotonic_increasing

    def test_total_rows_preserved(self, sample_df, cfg_data):
        fe = engineer_features(sample_df, cfg_data)
        train, val, test = split(fe)
        assert len(train) + len(val) + len(test) == len(fe)


class TestImputeBessSoc:
    """Tests for BESS SoC re-simulation logic."""

    def test_constant_soc_triggers_resimulation(self, cfg_data):
        """If SoC column is constant, it should be re-simulated."""
        rng = np.random.default_rng(7)
        idx = pd.date_range("2025-01-01", periods=100, freq="h")
        # Use new names to avoid renaming logic in test
        df = pd.DataFrame({
            "tao_load_mw": rng.uniform(5, 10, 100),
            "capacity_mw": rng.uniform(8, 16, 100),
            "bess_soc_pct": np.full(100, 65.0),  # constant → triggers resim
        }, index=idx)
        result = impute_bess_soc(df, cfg_data)
        # After resim, SoC should NOT be constant
        assert result["bess_soc_pct"].std() > 0.1

    def test_variable_soc_unchanged(self, cfg_data):
        """If SoC already varies, it should be left alone."""
        rng = np.random.default_rng(7)
        idx = pd.date_range("2025-01-01", periods=100, freq="h")
        soc_values = rng.uniform(20, 95, 100)
        df = pd.DataFrame({
            "tao_load_mw": rng.uniform(5, 10, 100),
            "capacity_mw": rng.uniform(8, 16, 100),
            "bess_soc_pct": soc_values,
        }, index=idx)
        result = impute_bess_soc(df, cfg_data)
        np.testing.assert_array_equal(result["bess_soc_pct"].values, soc_values)

    def test_resimulated_soc_within_bounds(self, cfg_data):
        """Re-simulated SoC must stay within config soc_min/soc_max."""
        rng = np.random.default_rng(7)
        idx = pd.date_range("2025-01-01", periods=200, freq="h")
        df = pd.DataFrame({
            "tao_load_mw": rng.uniform(5, 10, 200),
            "capacity_mw": rng.uniform(8, 16, 200),
            "bess_soc_pct": np.full(200, 65.0),
        }, index=idx)
        result = impute_bess_soc(df, cfg_data)
        bc = cfg_data["bess"]
        assert result["bess_soc_pct"].min() >= bc["soc_min"] * 100 - 0.1
        assert result["bess_soc_pct"].max() <= bc["soc_max"] * 100 + 0.1
