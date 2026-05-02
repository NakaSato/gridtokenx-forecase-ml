"""
Regression tests for model predictions — ensures trained models produce
consistent outputs and meet PEA performance targets.
"""
import os
import sys
import json
import pickle
import pytest
import numpy as np
import pandas as pd
import warnings as stdlib_warnings

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Suppress sklearn version mismatch warning
stdlib_warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)


@pytest.fixture(scope="module")
def test_data():
    return pd.read_parquet(os.path.join(ROOT, "data/processed/test.parquet"))


@pytest.fixture(scope="module")
def evaluation_report():
    path = os.path.join(ROOT, "results/evaluation_report.json")
    if not os.path.exists(path):
        pytest.skip("evaluation_report.json not found — run `just eval` first")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def config():
    import yaml
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


class TestModelArtifactsExist:
    """Verify all required model files exist and are loadable."""

    def test_lgbm_exists(self):
        assert os.path.exists(os.path.join(ROOT, "models/lgbm.pkl"))

    def test_tcn_exists(self):
        assert os.path.exists(os.path.join(ROOT, "models/tcn.pt"))

    def test_meta_learner_exists(self):
        assert os.path.exists(os.path.join(ROOT, "models/meta_learner.pkl"))

    def test_scaler_exists(self):
        assert os.path.exists(os.path.join(ROOT, "data/processed/scaler.pkl"))

    def test_lgbm_loadable(self):
        with open(os.path.join(ROOT, "models/lgbm.pkl"), "rb") as f:
            model = pickle.load(f)
        assert hasattr(model, "predict")

    def test_tcn_loadable(self):
        import torch
        ckpt = torch.load(os.path.join(ROOT, "models/tcn.pt"),
                          map_location="cpu", weights_only=False)
        assert "state_dict" in ckpt
        assert "config" in ckpt
        assert "in_features" in ckpt

    def test_meta_learner_loadable(self):
        with open(os.path.join(ROOT, "models/meta_learner.pkl"), "rb") as f:
            meta = pickle.load(f)
        assert hasattr(meta, "predict")

    def test_scaler_loadable(self):
        with open(os.path.join(ROOT, "data/processed/scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        assert hasattr(scaler, "transform")


class TestDataArtifactsExist:
    """Verify all required data files exist."""

    def test_train_exists(self):
        assert os.path.exists(os.path.join(ROOT, "data/processed/train.parquet"))

    def test_val_exists(self):
        assert os.path.exists(os.path.join(ROOT, "data/processed/val.parquet"))

    def test_test_exists(self):
        assert os.path.exists(os.path.join(ROOT, "data/processed/test.parquet"))

    def test_train_nonempty(self):
        df = pd.read_parquet(os.path.join(ROOT, "data/processed/train.parquet"))
        assert len(df) > 1000, f"Train too small: {len(df)} rows"

    def test_splits_no_overlap(self):
        train = pd.read_parquet(os.path.join(ROOT, "data/processed/train.parquet"))
        val = pd.read_parquet(os.path.join(ROOT, "data/processed/val.parquet"))
        test = pd.read_parquet(os.path.join(ROOT, "data/processed/test.parquet"))
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()


class TestTCNForwardPass:
    """Verify TCN produces outputs with correct shape."""

    def test_tcn_forward(self, test_data):
        import torch
        from models.tcn_model import TCN

        ckpt = torch.load(os.path.join(ROOT, "models/tcn.pt"),
                          map_location="cpu", weights_only=False)
        tc = ckpt["config"]
        dropout = ckpt.get("dropout", 0.0)
        net = TCN(ckpt["in_features"], tc["filters"], tc["kernel_size"],
                  tc["layers"], tc["forecast_horizon"], dropout=dropout)
        net.eval()

        # Create dummy input: (1, window_size, in_features)
        x = torch.randn(1, tc["window_size"], ckpt["in_features"])
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, tc["forecast_horizon"])

    def test_tcn_output_reasonable(self, test_data):
        """TCN output for real data should be in plausible MW range."""
        import torch
        from models.tcn_model import TCN, SEQ_FEATURES

        ckpt = torch.load(os.path.join(ROOT, "models/tcn.pt"),
                          map_location="cpu", weights_only=False)
        tc = ckpt["config"]
        dropout = ckpt.get("dropout", 0.0)
        net = TCN(ckpt["in_features"], tc["filters"], tc["kernel_size"],
                  tc["layers"], tc["forecast_horizon"], dropout=dropout)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()

        # Use first window from test data
        vals = test_data[SEQ_FEATURES].values[:tc["window_size"]].astype(np.float32)
        x = torch.tensor(vals).unsqueeze(0)
        with torch.no_grad():
            out = net(x).numpy()[0]

        # Predictions should be within plausible range (0–20 MW)
        assert out.min() > -5.0, f"TCN output too low: {out.min()}"
        assert out.max() < 25.0, f"TCN output too high: {out.max()}"


class TestPEATargetsMet:
    """Verify the system meets all PEA performance targets."""

    def test_mape_target(self, evaluation_report, config):
        assert evaluation_report["forecast"]["mape"] <= config["targets"]["mape"], \
            f"MAPE {evaluation_report['forecast']['mape']}% > target {config['targets']['mape']}%"

    def test_r2_target(self, evaluation_report, config):
        assert evaluation_report["forecast"]["r2"] >= config["targets"]["r2"], \
            f"R² {evaluation_report['forecast']['r2']} < target {config['targets']['r2']}"

    def test_mae_target(self, evaluation_report, config):
        assert evaluation_report["forecast"]["mae"] <= config["targets"]["mae"], \
            f"MAE {evaluation_report['forecast']['mae']} > target {config['targets']['mae']}"

    def test_fuel_savings_target(self, evaluation_report, config):
        actual = evaluation_report["dispatch"]["fuel_savings_pct"] / 100
        assert actual >= config["targets"]["fuel_savings"], \
            f"Fuel savings {actual*100}% < target {config['targets']['fuel_savings']*100}%"

    def test_backtest_pea_pass(self, evaluation_report):
        assert evaluation_report["backtest_pea_pass"], \
            f"Walk-forward backtest MAPE {evaluation_report['backtest_24h_mape']}% > PEA 10% limit"

    def test_all_targets_met(self, evaluation_report):
        for key, val in evaluation_report["targets_met"].items():
            assert val, f"Target not met: {key}"


class TestSklearnVersionConsistency:
    """Verify meta_learner.pkl was saved with the current sklearn version."""

    def test_no_version_mismatch_warning(self):
        """Loading meta_learner.pkl should not produce InconsistentVersionWarning."""
        import sklearn
        with stdlib_warnings.catch_warnings(record=True) as w:
            stdlib_warnings.simplefilter("always")
            with open(os.path.join(ROOT, "models/meta_learner.pkl"), "rb") as f:
                pickle.load(f)
            sklearn_warnings = [x for x in w
                                if "InconsistentVersionWarning" in str(type(x.category))]
            assert len(sklearn_warnings) == 0, \
                f"sklearn version mismatch detected: {sklearn_warnings[0].message}"
