# Testing Infrastructure

This directory contains comprehensive testing infrastructure for the Privacy-Preserving Agent Fine-Tuner.

## Test Organization

### Directory Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── README.md               # This file
├── config/                 # Test configuration files
│   └── test_configs.yaml  # Test scenario configurations
├── data/                   # Test data and fixtures
│   └── sample_datasets.json # Sample datasets for testing
├── fixtures/               # Test fixtures and utilities
│   ├── privacy_configs.py  # Privacy configuration fixtures
│   └── test_data.py        # Test data generators
├── unit/                   # Unit tests
│   ├── test_context_guard.py
│   └── test_privacy_config.py
├── integration/            # Integration tests
│   └── test_end_to_end_privacy.py
├── privacy/                # Privacy-specific tests
│   └── test_differential_privacy.py
├── security/               # Security tests
│   └── test_privacy_attacks.py
├── performance/            # Performance tests
│   └── test_privacy_overhead.py
└── utils/                  # Test utilities
    └── test_helpers.py
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Fast, isolated tests for individual components
- Mock external dependencies
- High code coverage focus
- Run with: `pytest tests/unit/`

### Integration Tests (`tests/integration/`)
- End-to-end testing of component interactions
- Real or containerized external services
- Privacy guarantee validation
- Run with: `pytest tests/integration/`

### Privacy Tests (`tests/privacy/`)
- Differential privacy guarantee validation
- Privacy budget consumption testing
- Privacy accounting verification
- Run with: `pytest tests/privacy/`

### Security Tests (`tests/security/`)
- Privacy attack simulation (membership inference, model inversion)
- Defense mechanism validation
- Secure computation testing
- Run with: `pytest tests/security/`

### Performance Tests (`tests/performance/`)
- Training speed and throughput
- Memory usage profiling
- Scalability testing
- Run with: `pytest tests/performance/`

## Test Markers

Use pytest markers to categorize and run specific test types:

```bash
# Run only unit tests
pytest -m unit

# Run privacy-related tests
pytest -m privacy

# Run tests requiring GPU
pytest -m gpu

# Skip slow tests
pytest -m "not slow"

# Run security tests
pytest -m security

# Run compliance tests
pytest -m compliance

# Run federated learning tests
pytest -m federated
```

## Available Markers

- `unit`: Unit tests
- `integration`: Integration tests  
- `privacy`: Privacy guarantee tests
- `security`: Security-related tests
- `performance`: Performance tests
- `compliance`: Compliance tests (GDPR, HIPAA, CCPA)
- `federated`: Federated learning tests
- `enclave`: Secure enclave tests
- `gpu`: Tests requiring GPU
- `slow`: Slow-running tests

## Configuration

### Test Environment Variables

The test suite automatically sets up a test environment with:

```bash
ENVIRONMENT=test
DEBUG=true
DATABASE_URL=sqlite:///:memory:
REDIS_URL=redis://localhost:6379/15
PRIVACY_EPSILON=1.0
PRIVACY_DELTA=1e-5
```

### Test Configurations

Use test configurations from `tests/config/test_configs.yaml`:

```python
# In your test
@pytest.fixture
def privacy_config():
    from tests.utils.config_loader import load_test_config
    return load_test_config("privacy_configs.moderate")
```

## Common Test Patterns

### Testing Privacy Guarantees

```python
@pytest.mark.privacy
def test_differential_privacy_guarantee(privacy_config, trainer):
    # Test that privacy budget is properly consumed
    initial_budget = trainer.privacy_monitor.remaining_budget
    trainer.train_step(batch)
    final_budget = trainer.privacy_monitor.remaining_budget
    
    assert final_budget < initial_budget
    assert trainer.privacy_monitor.validate_privacy_guarantee()
```

### Testing Security Defenses

```python
@pytest.mark.security
def test_membership_inference_defense(trained_model, attack_data):
    # Simulate membership inference attack
    attack_result = run_membership_inference_attack(
        model=trained_model,
        attack_data=attack_data
    )
    
    # Verify attack success rate is below threshold
    assert attack_result.success_rate < 0.55  # Near random chance
```

### Testing Performance

```python
@pytest.mark.performance
@pytest.mark.slow
def test_training_throughput(large_dataset, trainer):
    start_time = time.time()
    trainer.train(large_dataset, epochs=1)
    duration = time.time() - start_time
    
    throughput = len(large_dataset) / duration
    assert throughput > 100  # samples per second
```

### Testing Compliance

```python
@pytest.mark.compliance
def test_gdpr_compliance(gdpr_config, processor):
    # Test GDPR requirements
    processor.configure(gdpr_config)
    
    assert processor.supports_right_to_be_forgotten()
    assert processor.provides_data_portability()
    assert processor.logs_data_processing()
```

## Running Tests

### Complete Test Suite

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=privacy_finetuner --cov-report=html

# Run in parallel
pytest -n auto
```

### Specific Test Categories

```bash
# Fast tests only (unit tests)
pytest tests/unit/ -v

# Privacy tests with detailed output
pytest tests/privacy/ -v -s

# Security tests (may be slow)
pytest tests/security/ --tb=long

# Performance tests (requires more time)
pytest tests/performance/ -v --durations=10
```

### CI/CD Integration

```bash
# Pre-commit testing (fast)
pytest tests/unit/ tests/privacy/ -x --tb=short

# Full CI testing
pytest --cov=privacy_finetuner --cov-fail-under=80 --junitxml=junit.xml

# Nightly testing (includes slow tests)
pytest tests/ --cov=privacy_finetuner --cov-report=xml
```

## Test Data Management

### Fixtures and Mock Data

- Use `conftest.py` fixtures for common test objects
- Store test data in `tests/data/` for reusability
- Use factories for generating test data variations

### Privacy-Safe Testing

- Never use real sensitive data in tests
- Use synthetic data that matches real data patterns
- Validate that test data doesn't leak into logs

### Performance Test Data

- Use appropriately sized datasets for performance tests
- Consider memory constraints in CI environments
- Profile memory usage to prevent OOM errors

## Debugging Tests

### Logging in Tests

```python
import logging
logging.getLogger("privacy_finetuner").setLevel(logging.DEBUG)
```

### Interactive Debugging

```python
# Add breakpoint in test
import pdb; pdb.set_trace()

# Or use pytest's built-in debugger
pytest --pdb tests/unit/test_specific.py::test_function
```

### Test Output Capture

```bash
# Show print statements
pytest -s

# Show detailed output
pytest -v -s

# Show only failed test output
pytest --tb=short
```

## Contributing

### Adding New Tests

1. Choose appropriate test category directory
2. Add relevant pytest markers
3. Use existing fixtures when possible
4. Follow naming conventions (`test_*.py`, `test_*()`)
5. Add docstrings explaining test purpose

### Test Quality Guidelines

- Tests should be fast, reliable, and isolated
- Use descriptive test names and docstrings  
- Mock external dependencies appropriately
- Test both success and failure scenarios
- Validate privacy guarantees in privacy tests

### Privacy Testing Best Practices

- Always test privacy budget consumption
- Validate differential privacy parameters
- Test privacy guarantee preservation under various conditions
- Simulate realistic attack scenarios
- Verify defense mechanisms work as expected