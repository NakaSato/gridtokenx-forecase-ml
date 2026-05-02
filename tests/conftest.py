"""
Shared pytest fixtures for GridTokenX test suite.
"""
import os
import sys
import pytest
import numpy as np
import yaml

# Ensure project root is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@pytest.fixture(scope="session")
def cfg():
    """Load project config.yaml once per test session."""
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def sample_load_24h():
    """Typical 24h load profile (MW) — realistic Ko Tao range."""
    rng = np.random.default_rng(42)
    base = 7.5 + 1.5 * np.sin(np.linspace(0, 2 * np.pi, 24))  # diurnal
    noise = rng.normal(0, 0.2, 24)
    return np.clip(base + noise, 5.0, 10.0)


@pytest.fixture(scope="session")
def sample_circuit_24h():
    """24h circuit capacity — includes evening bottleneck window."""
    circuit = np.full(24, 14.0)
    # Bottleneck at 18:00–21:00
    circuit[18:22] = np.array([3.0, 2.5, 3.5, 4.0])
    return circuit


@pytest.fixture(scope="session")
def sample_circuit_no_bottleneck():
    """24h circuit capacity — no bottleneck (all surplus)."""
    return np.full(24, 16.0)


@pytest.fixture(scope="session")
def high_deficit_load():
    """24h load that exceeds circuit + BESS capacity."""
    return np.full(24, 12.0)


@pytest.fixture(scope="session")
def low_circuit():
    """24h circuit with severe bottleneck."""
    return np.full(24, 2.0)
