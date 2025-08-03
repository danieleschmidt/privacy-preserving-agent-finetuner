"""Pytest configuration and shared fixtures."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

# Test configuration
TEST_MODEL_NAME = "microsoft/DialoGPT-small"
TEST_EPSILON = 1.0
TEST_DELTA = 1e-5
TEST_BATCH_SIZE = 2
TEST_MAX_LENGTH = 128


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def test_config_dir() -> Path:
    """Get the test config directory."""
    return Path(__file__).parent / "config"


@pytest.fixture
def privacy_config():
    """Create a test privacy configuration."""
    from privacy_finetuner.core.privacy_config import PrivacyConfig
    
    return PrivacyConfig(
        epsilon=TEST_EPSILON,
        delta=TEST_DELTA,
        max_grad_norm=1.0,
        noise_multiplier=0.5,
        accounting_mode="rdp"
    )


@pytest.fixture
def model_config():
    """Create a test model configuration."""
    from privacy_finetuner.config.model import ModelConfig
    
    return ModelConfig(
        model_name=TEST_MODEL_NAME,
        max_length=TEST_MAX_LENGTH,
        batch_size=TEST_BATCH_SIZE,
        learning_rate=5e-5,
        warmup_steps=10,
        max_steps=50,
        gradient_accumulation_steps=1,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )


@pytest.fixture
def training_config():
    """Create a test training configuration."""
    from privacy_finetuner.config.training import TrainingConfig
    
    return TrainingConfig(
        epochs=2,
        batch_size=TEST_BATCH_SIZE,
        learning_rate=5e-5,
        warmup_steps=10,
        max_steps=50,
        gradient_accumulation_steps=1,
        save_steps=25,
        eval_steps=25,
        logging_steps=10,
        dataloader_num_workers=0,
        fp16=False,
        seed=42,
    )


@pytest.fixture
def test_tokenizer():
    """Create a test tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def test_model_config():
    """Create a test model config from HuggingFace."""
    return AutoConfig.from_pretrained(TEST_MODEL_NAME)


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    return [
        {"text": "This is a sample training text for privacy testing."},
        {"text": "Another example with different content for testing."},
        {"text": "Privacy-preserving machine learning is important."},
        {"text": "Differential privacy provides formal guarantees."},
        {"text": "Federated learning enables distributed training."},
    ]


@pytest.fixture
def sample_sensitive_data():
    """Create sample data with sensitive information."""
    return [
        {"text": "John Doe's SSN is 123-45-6789 and phone is 555-123-4567"},
        {"text": "Email alice@example.com and credit card 4111-1111-1111-1111"},
        {"text": "Patient ID 12345 has condition diabetes"},
        {"text": "User Bob Smith lives at 123 Main St, Anytown USA"},
        {"text": "Account number 987654321 has balance $1000"},
    ]


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    with patch("redis.Redis") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_database():
    """Mock database session."""
    with patch("privacy_finetuner.database.connection.get_db_session") as mock:
        session = MagicMock()
        mock.return_value = session
        yield session


@pytest.fixture
def test_database():
    """Create test database session."""
    from privacy_finetuner.database.connection import DatabaseManager
    from privacy_finetuner.database.models import Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()


@pytest.fixture
def mock_s3():
    """Mock S3 client."""
    with patch("boto3.client") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_huggingface_hub():
    """Mock HuggingFace Hub."""
    with patch("huggingface_hub.HfApi") as mock:
        api = MagicMock()
        mock.return_value = api
        yield api


@pytest.fixture
def privacy_budget_monitor():
    """Create a privacy budget monitor."""
    from privacy_finetuner.privacy.budget_monitor import PrivacyBudgetMonitor
    
    return PrivacyBudgetMonitor(
        total_epsilon=TEST_EPSILON,
        total_delta=TEST_DELTA,
        accounting_mode="rdp",
    )


@pytest.fixture
def context_guard():
    """Create a context guard for testing."""
    from privacy_finetuner.context.guard import ContextGuard
    from privacy_finetuner.context.strategies import (
        PIIRemovalStrategy,
        EntityHashingStrategy,
    )
    
    strategies = [
        PIIRemovalStrategy(),
        EntityHashingStrategy(salt="test-salt"),
    ]
    
    return ContextGuard(strategies=strategies)


@pytest.fixture
def mock_sgx_enclave():
    """Mock SGX enclave."""
    with patch("privacy_finetuner.secure_compute.sgx.SGXEnclave") as mock:
        enclave = MagicMock()
        mock.return_value = enclave
        enclave.is_available.return_value = True
        enclave.attest.return_value = b"mock_attestation"
        yield enclave


@pytest.fixture
def mock_nitro_enclave():
    """Mock AWS Nitro enclave."""
    with patch("privacy_finetuner.secure_compute.nitro.NitroEnclave") as mock:
        enclave = MagicMock()
        mock.return_value = enclave
        enclave.is_available.return_value = True
        enclave.attest.return_value = b"mock_attestation"
        yield enclave


@pytest.fixture
def federated_client_config():
    """Create federated learning client configuration."""
    from privacy_finetuner.federated.config import FederatedClientConfig
    
    return FederatedClientConfig(
        client_id="test-client-001",
        server_url="http://localhost:8080",
        aggregation_method="secure_sum",
        local_epochs=2,
        batch_size=TEST_BATCH_SIZE,
        privacy_budget=0.1,
    )


@pytest.fixture
def federated_server_config():
    """Create federated learning server configuration."""
    from privacy_finetuner.federated.config import FederatedServerConfig
    
    return FederatedServerConfig(
        min_clients=2,
        max_clients=10,
        rounds=5,
        aggregation_method="secure_sum",
        privacy_budget_per_round=0.2,
        byzantine_tolerance=0.3,
    )


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables."""
    test_env = {
        "ENVIRONMENT": "test",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "DATABASE_URL": "sqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6379/15",
        "PRIVACY_EPSILON": str(TEST_EPSILON),
        "PRIVACY_DELTA": str(TEST_DELTA),
        "MODEL_CACHE_DIR": "/tmp/test_models",
        "TRANSFORMERS_CACHE": "/tmp/test_models",
        "HF_HOME": "/tmp/test_models",
        "TORCH_HOME": "/tmp/test_models",
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


@pytest.fixture
def mock_wandb():
    """Mock Weights & Biases."""
    with patch("wandb.init") as mock_init, \
         patch("wandb.log") as mock_log, \
         patch("wandb.finish") as mock_finish:
        yield {
            "init": mock_init,
            "log": mock_log,
            "finish": mock_finish,
        }


@pytest.fixture
def mock_mlflow():
    """Mock MLflow."""
    with patch("mlflow.start_run") as mock_start, \
         patch("mlflow.log_metric") as mock_metric, \
         patch("mlflow.log_param") as mock_param, \
         patch("mlflow.end_run") as mock_end:
        yield {
            "start_run": mock_start,
            "log_metric": mock_metric,
            "log_param": mock_param,
            "end_run": mock_end,
        }


@pytest.fixture
async def async_client():
    """Create async HTTP client for testing."""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        yield session


@pytest.fixture
def performance_test_data():
    """Create larger dataset for performance testing."""
    return [
        {"text": f"This is test sentence number {i} for performance testing."}
        for i in range(1000)
    ]


@pytest.fixture
def compliance_test_data():
    """Create data for compliance testing."""
    return {
        "gdpr": [
            {"text": "EU citizen data for GDPR compliance testing"},
            {"text": "Personal data processing under GDPR"},
        ],
        "hipaa": [
            {"text": "Protected health information for HIPAA testing"},
            {"text": "Medical records data privacy testing"},
        ],
        "ccpa": [
            {"text": "California consumer data for CCPA testing"},
            {"text": "Personal information under CCPA"},
        ],
    }


# Markers for test organization
pytestmark = [
    pytest.mark.asyncio,
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "privacy: Privacy guarantee tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "security: Security-related tests")
    config.addinivalue_line("markers", "compliance: Compliance tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "federated: Federated learning tests")
    config.addinivalue_line("markers", "enclave: Secure enclave tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    for item in items:
        # Add markers based on test file path
        test_path = str(item.fspath)
        
        if "/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "/privacy/" in test_path:
            item.add_marker(pytest.mark.privacy)
        elif "/security/" in test_path:
            item.add_marker(pytest.mark.security)
        elif "/compliance/" in test_path:
            item.add_marker(pytest.mark.compliance)
        elif "/performance/" in test_path:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "/federated/" in test_path:
            item.add_marker(pytest.mark.federated)
        elif "/enclave/" in test_path:
            item.add_marker(pytest.mark.enclave)
        
        # Add GPU marker for tests requiring CUDA
        if "gpu" in item.name.lower() or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    """Clean up test artifacts after test session."""
    yield
    
    # Clean up temporary files
    import shutil
    test_dirs = [
        "/tmp/test_models",
        "/tmp/privacy_test_cache",
        "/tmp/test_logs",
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)