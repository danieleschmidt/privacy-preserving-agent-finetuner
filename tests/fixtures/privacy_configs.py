"""Test fixtures for privacy configuration testing."""

import pytest
from privacy_finetuner.core.privacy_config import PrivacyConfig


@pytest.fixture
def basic_privacy_config():
    """Basic privacy configuration for testing."""
    return PrivacyConfig(
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=0.5
    )


@pytest.fixture
def strict_privacy_config():
    """Strict privacy configuration with low epsilon."""
    return PrivacyConfig(
        epsilon=0.1,
        delta=1e-6,
        max_grad_norm=0.5,
        noise_multiplier=2.0
    )


@pytest.fixture
def relaxed_privacy_config():
    """Relaxed privacy configuration with higher epsilon."""
    return PrivacyConfig(
        epsilon=10.0,
        delta=1e-4,
        max_grad_norm=2.0,
        noise_multiplier=0.1
    )


@pytest.fixture
def federated_privacy_config():
    """Privacy configuration for federated learning."""
    return PrivacyConfig(
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=0.5,
        federated=True,
        min_clients=5,
        aggregation_method="secure_sum"
    )


@pytest.fixture
def privacy_config_params():
    """Parametrized privacy configurations for testing."""
    return [
        {"epsilon": 0.1, "delta": 1e-6, "expected_noise": "high"},
        {"epsilon": 1.0, "delta": 1e-5, "expected_noise": "medium"},
        {"epsilon": 10.0, "delta": 1e-4, "expected_noise": "low"},
    ]


@pytest.fixture
def invalid_privacy_configs():
    """Invalid privacy configurations for error testing."""
    return [
        {"epsilon": -1.0, "delta": 1e-5},  # Negative epsilon
        {"epsilon": 1.0, "delta": -1e-5},  # Negative delta
        {"epsilon": 0, "delta": 1e-5},     # Zero epsilon
        {"epsilon": 1.0, "delta": 0},      # Zero delta
        {"epsilon": float('inf'), "delta": 1e-5},  # Infinite epsilon
    ]