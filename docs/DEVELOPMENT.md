# Development Guide

## Quick Setup

1. **Install Poetry**: `curl -sSL https://install.python-poetry.org | python3 -`
2. **Install dependencies**: `make dev-install`
3. **Setup pre-commit**: `poetry run pre-commit install`
4. **Run tests**: `make test`

## Development Environment

### Requirements
- Python 3.9+
- Poetry package manager
- Docker (optional)
- CUDA 11.8+ (GPU support)

### Local Setup
```bash
# Clone and setup
git clone <repository-url>
cd privacy-preserving-agent-finetuner
make dev-install

# Environment configuration
cp .env.example .env
# Edit .env with your settings
```

### Docker Development
```bash
make docker-build-dev
make docker-run-dev
```

## Testing

Run test suites with appropriate markers:
- `make test` - All tests
- `poetry run pytest -m unit` - Unit tests only
- `poetry run pytest -m privacy` - Privacy-specific tests
- `poetry run pytest --cov` - With coverage

## Code Quality

Pre-commit hooks run automatically:
- Black code formatting
- isort import sorting
- flake8 linting
- mypy type checking
- Security scanning with bandit

Manual execution: `poetry run pre-commit run --all-files`

## Privacy & Security

Always follow these practices:
- Never log sensitive data
- Validate privacy parameters (ε, δ)
- Use synthetic data for testing
- Review security implications

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.