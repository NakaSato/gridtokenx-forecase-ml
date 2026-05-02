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
    """Create a minimal synthetic DataFrame matching ko_tao_grid.parquet schema."""
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
        for lag in [1, 24, 168]:
            assert f"Load_Lag_{lag}h" in result.columns, f"Missing Load_Lag_{lag}h"

    def test_rolling_features_created(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        for w in [3, 6]:
            assert f"Load_Roll_Mean_{w}h" in result.columns
            assert f"Load_Roll_Std_{w}h" in result.columns

    def test_heat_index_correct(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        assert "Heat_Index" in result.columns
        # Heat index = temp * humidity / 100
        expected = result["Dry_Bulb_Temp"] * result["Rel_Humidity"] / 100
        np.testing.assert_allclose(result["Heat_Index"].values, expected.values, rtol=1e-5)

    def test_time_features_created(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        assert "Hour_of_Day" in result.columns
        assert "Day_of_Week" in result.columns
        assert "Is_High_Season" in result.columns

    def test_hour_of_day_range(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        assert result["Hour_of_Day"].min() >= 0
        assert result["Hour_of_Day"].max() <= 23

    def test_is_high_season_binary(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        assert set(result["Is_High_Season"].unique()).issubset({0, 1})

    def test_leaky_features_dropped(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        # Net_Delta_MW and Circuit_Cap_MW should be dropped (leaky)
        assert "Net_Delta_MW" not in result.columns
        assert "Circuit_Cap_MW" not in result.columns
        assert "Is_Weekend" not in result.columns

    def test_no_nan_after_dropna(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        assert result.isna().sum().sum() == 0, "NaN values remain after dropna"

    def test_tourist_index_fallback(self, sample_df, cfg_data):
        """If Tourist_Index is missing, it should be derived from Is_High_Season."""
        df = sample_df.drop(columns=["Tourist_Index"])
        result = engineer_features(df, cfg_data)
        assert "Tourist_Index" in result.columns

    def test_weather_trends_created(self, sample_df, cfg_data):
        result = engineer_features(sample_df, cfg_data)
        assert "Temp_Gradient" in result.columns
        for w in [3, 6]:
            assert f"Temp_Roll_Mean_{w}h" in result.columns
            assert f"Humid_Roll_Mean_{w}h" in result.columns

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
        assert abs(len(train) / total - 0.70) < 0.02
        assert abs(len(val) / total - 0.15) < 0.02
        assert abs(len(test) / total - 0.15) < 0.02

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
        df = pd.DataFrame({
            "Island_Load_MW": rng.uniform(5, 10, 100),
            "Circuit_Cap_MW": rng.uniform(8, 16, 100),
            "BESS_SoC_Pct": np.full(100, 65.0),  # constant → triggers resim
        }, index=idx)
        result = impute_bess_soc(df, cfg_data)
        # After resim, SoC should NOT be constant
        assert result["BESS_SoC_Pct"].std() > 0.1

    def test_variable_soc_unchanged(self, cfg_data):
        """If SoC already varies, it should be left alone."""
        rng = np.random.default_rng(7)
        idx = pd.date_range("2025-01-01", periods=100, freq="h")
        soc_values = rng.uniform(20, 95, 100)
        df = pd.DataFrame({
            "Island_Load_MW": rng.uniform(5, 10, 100),
            "Circuit_Cap_MW": rng.uniform(8, 16, 100),
            "BESS_SoC_Pct": soc_values,
        }, index=idx)
        result = impute_bess_soc(df, cfg_data)
        np.testing.assert_array_equal(result["BESS_SoC_Pct"].values, soc_values)

    def test_resimulated_soc_within_bounds(self, cfg_data):
        """Re-simulated SoC must stay within config soc_min/soc_max."""
        rng = np.random.default_rng(7)
        idx = pd.date_range("2025-01-01", periods=200, freq="h")
        df = pd.DataFrame({
            "Island_Load_MW": rng.uniform(5, 10, 200),
            "Circuit_Cap_MW": rng.uniform(8, 16, 200),
            "BESS_SoC_Pct": np.full(200, 65.0),
        }, index=idx)
        result = impute_bess_soc(df, cfg_data)
        bc = cfg_data["bess"]
        assert result["BESS_SoC_Pct"].min() >= bc["soc_min"] * 100 - 0.1
        assert result["BESS_SoC_Pct"].max() <= bc["soc_max"] * 100 + 0.1
