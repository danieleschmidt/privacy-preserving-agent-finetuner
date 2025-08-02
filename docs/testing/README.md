# Testing Strategy

This document outlines the comprehensive testing strategy for the Privacy-Preserving Agent Fine-Tuner.

## Test Organization

### Test Categories

- **Unit Tests** (`tests/unit/`): Test individual components in isolation
- **Integration Tests** (`tests/integration/`): Test component interactions
- **End-to-End Tests** (`tests/e2e/`): Test complete workflows
- **Privacy Tests** (`tests/privacy/`): Test differential privacy guarantees
- **Security Tests** (`tests/security/`): Test security mechanisms and attack resistance
- **Performance Tests** (`tests/performance/`): Test performance and scalability
- **Compliance Tests** (`tests/compliance/`): Test regulatory compliance (GDPR, HIPAA, etc.)
- **Federated Learning Tests** (`tests/federated/`): Test federated learning functionality
- **Enclave Tests** (`tests/enclave/`): Test secure enclave integration

### Test Markers

```bash
# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m privacy      # Privacy guarantee tests
pytest -m security     # Security tests
pytest -m compliance   # Compliance tests
pytest -m federated    # Federated learning tests
pytest -m enclave      # Secure enclave tests
pytest -m performance  # Performance tests
pytest -m slow         # Slow-running tests
pytest -m gpu          # GPU-requiring tests

# Exclude specific categories
pytest -m "not slow"   # Skip slow tests
pytest -m "not gpu"    # Skip GPU tests
```

## Running Tests

### Quick Test Suite

```bash
# Run fast unit tests
make test-unit

# Run all tests with coverage
make test

# Run specific test category
pytest tests/privacy/ -v
```

### Full Test Suite

```bash
# Run all checks (lint, test, security, privacy)
make check

# Run CI pipeline locally
make ci
```

### Privacy-Specific Testing

```bash
# Run privacy guarantee tests
make test-privacy

# Run with specific privacy budget
pytest tests/privacy/ --privacy-budget=1.0

# Run privacy compliance checks
make privacy-check
```

## Test Configuration

### Environment Variables

Tests automatically set up appropriate environment variables via `conftest.py`:

- `ENVIRONMENT=test`
- `DEBUG=true`
- `DATABASE_URL=sqlite:///:memory:`
- `REDIS_URL=redis://localhost:6379/15`

### Fixtures

Comprehensive fixtures are available in `conftest.py`:

- **Configuration fixtures**: `privacy_config`, `model_config`, `training_config`
- **Data fixtures**: `sample_training_data`, `sample_sensitive_data`
- **Mock fixtures**: `mock_redis`, `mock_database`, `mock_s3`
- **Hardware fixtures**: `mock_sgx_enclave`, `mock_nitro_enclave`
- **Federated fixtures**: `federated_client_config`, `federated_server_config`

## Privacy Testing Strategy

### Differential Privacy Validation

1. **Privacy Budget Tracking**: Verify epsilon and delta consumption
2. **Noise Validation**: Test that appropriate noise is added
3. **Composition Analysis**: Test privacy budget composition across iterations
4. **Attack Resistance**: Test against membership inference and model inversion attacks

### Privacy Guarantee Tests

```python
@pytest.mark.privacy
def test_privacy_guarantee(privacy_config):
    """Test that differential privacy guarantees are maintained."""
    # Test implementation validates formal privacy guarantees
```

## Security Testing Strategy

### Attack Simulation

1. **Membership Inference Attacks**: Test resistance to data membership inference
2. **Model Inversion Attacks**: Test resistance to input reconstruction
3. **Property Inference Attacks**: Test resistance to property inference
4. **Model Extraction Attacks**: Test resistance to model stealing

### Security Test Examples

```python
@pytest.mark.security
def test_membership_inference_resistance():
    """Test resistance to membership inference attacks."""
    # Implementation tests attack resistance
```

## Compliance Testing Strategy

### Regulatory Frameworks

1. **GDPR**: Data subject rights, consent management, privacy by design
2. **HIPAA**: Healthcare data protection, access controls, audit trails
3. **CCPA**: Consumer privacy rights, data transparency, opt-out mechanisms
4. **EU AI Act**: AI system requirements, risk assessments, documentation

### Compliance Test Examples

```python
@pytest.mark.compliance
def test_gdpr_data_subject_rights():
    """Test GDPR data subject rights implementation."""
    # Test right to access, rectification, erasure, portability
```

## Performance Testing Strategy

### Benchmarks

1. **Training Performance**: Training throughput with privacy mechanisms
2. **Memory Usage**: Memory overhead from privacy computations
3. **Scalability**: Performance scaling with dataset size and privacy budgets
4. **Latency**: Inference latency with privacy protections

### Performance Test Examples

```python
@pytest.mark.performance
@pytest.mark.slow
def test_training_performance_with_privacy():
    """Benchmark training performance with privacy mechanisms."""
    # Performance measurement and validation
```

## Continuous Integration

### Test Automation

The CI pipeline runs:

1. **Fast tests** on every commit
2. **Full test suite** on pull requests
3. **Performance tests** on releases
4. **Security scans** on every commit
5. **Compliance checks** on every commit

### Test Reports

- **Coverage reports**: HTML and XML coverage reports generated
- **Performance reports**: Benchmark results and trend analysis
- **Security reports**: Vulnerability and compliance scan results
- **Privacy reports**: Privacy guarantee validation results

## Best Practices

### Test Development

1. **Test-Driven Development**: Write tests before implementation
2. **Privacy-First Testing**: Include privacy tests for all data-handling code
3. **Security-Aware Testing**: Include security tests for all security-critical code
4. **Compliance Testing**: Include compliance tests for all regulated features

### Test Maintenance

1. **Regular Updates**: Keep tests updated with code changes
2. **Test Review**: Include test review in code review process
3. **Test Cleanup**: Remove obsolete tests and fix flaky tests
4. **Test Documentation**: Document complex test scenarios and requirements

## Troubleshooting

### Common Issues

1. **GPU Tests Failing**: Check CUDA availability with `torch.cuda.is_available()`
2. **Privacy Tests Failing**: Verify privacy budget configuration
3. **Slow Tests**: Use `-m "not slow"` to skip performance tests
4. **Flaky Tests**: Check for race conditions and proper test isolation

### Debugging

```bash
# Run with verbose output
pytest -v -s tests/

# Run specific test with debugging
pytest tests/unit/test_privacy_config.py::test_privacy_budget -vvv -s

# Run with coverage and HTML report
pytest --cov=privacy_finetuner --cov-report=html
```

## Contributing

When adding new features:

1. **Add unit tests** for all new functions and classes
2. **Add integration tests** for feature interactions
3. **Add privacy tests** if handling sensitive data
4. **Add security tests** if implementing security features
5. **Add compliance tests** if implementing regulated features
6. **Update documentation** for new test categories or procedures

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.
