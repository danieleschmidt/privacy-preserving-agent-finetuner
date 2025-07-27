# Contributing to Privacy-Preserving Agent Finetuner

Welcome! We're excited that you're interested in contributing to the Privacy-Preserving Agent Finetuner project. This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Development Guidelines](#development-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Privacy and Security Guidelines](#privacy-and-security-guidelines)
- [Code Review Process](#code-review-process)
- [Release Process](#release-process)
- [Community](#community)

## Code of Conduct

This project adheres to the [Contributor Covenant](https://www.contributor-covenant.org/). By participating, you are expected to uphold this code. Please report unacceptable behavior to [conduct@terragon-labs.com](mailto:conduct@terragon-labs.com).

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Docker (optional, for containerized development)
- CUDA 11.8+ (optional, for GPU support)

### First Steps

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/privacy-preserving-agent-finetuner.git
   cd privacy-preserving-agent-finetuner
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/terragon-labs/privacy-preserving-agent-finetuner.git
   ```

## Development Setup

### Local Development

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install dependencies**:
   ```bash
   make dev-install
   ```

3. **Set up pre-commit hooks**:
   ```bash
   poetry run pre-commit install
   ```

4. **Copy environment configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run tests** to verify setup:
   ```bash
   make test
   ```

### Docker Development

1. **Build development image**:
   ```bash
   make docker-build-dev
   ```

2. **Run development container**:
   ```bash
   make docker-run-dev
   ```

### VS Code Development

1. **Open in VS Code**:
   ```bash
   code .
   ```

2. **Use Dev Container** (recommended):
   - Install the "Remote - Containers" extension
   - Press `F1` and select "Remote-Containers: Reopen in Container"

## Contributing Process

### 1. Choose an Issue

- Look for issues labeled `good first issue` for beginners
- Check issues labeled `help wanted` for areas needing assistance
- For new features, create an issue first to discuss the approach

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Changes

- Follow the [development guidelines](#development-guidelines)
- Write tests for your changes
- Update documentation as needed
- Ensure all checks pass

### 4. Commit Changes

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add privacy budget monitoring"
git commit -m "fix: resolve memory leak in training loop"
git commit -m "docs: update API documentation"
```

### 5. Push and Create Pull Request

```bash
git push origin your-branch-name
```

Then create a pull request on GitHub with:
- Clear title and description
- Reference to related issues
- Screenshots or examples if applicable
- Checklist completion

## Development Guidelines

### Code Style

- **Python**: Follow PEP 8 and use type hints
- **Line length**: 88 characters (Black formatter)
- **Imports**: Use absolute imports, organize with isort
- **Docstrings**: Use Google-style docstrings

### Project Structure

```
privacy_finetuner/
├── api/                 # API endpoints and server
├── auth/               # Authentication and authorization
├── config/             # Configuration management
├── context/            # Context protection and guards
├── data/               # Data handling and processing
├── federated/          # Federated learning components
├── models/             # Model definitions and utilities
├── privacy/            # Privacy mechanisms and accounting
├── secure_compute/     # Secure computation (SGX, Nitro)
├── training/           # Training orchestration
└── utils/              # Utility functions
```

### Naming Conventions

- **Classes**: PascalCase (`PrivacyEngine`)
- **Functions/Variables**: snake_case (`calculate_privacy_loss`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_EPSILON`)
- **Files**: snake_case (`privacy_engine.py`)

### Error Handling

- Use specific exception types
- Provide meaningful error messages
- Log errors appropriately
- Handle privacy-related errors carefully

```python
from privacy_finetuner.exceptions import PrivacyBudgetExhaustedException

def consume_privacy_budget(epsilon: float) -> None:
    if epsilon > remaining_budget:
        raise PrivacyBudgetExhaustedException(
            f"Requested epsilon {epsilon} exceeds remaining budget {remaining_budget}"
        )
```

## Testing Guidelines

### Test Types

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **Privacy Tests**: Verify privacy guarantees
4. **Security Tests**: Test security measures
5. **Performance Tests**: Measure performance characteristics

### Writing Tests

```python
import pytest
from privacy_finetuner.privacy.engine import PrivacyEngine

@pytest.mark.unit
def test_privacy_engine_initialization():
    """Test privacy engine initializes correctly."""
    engine = PrivacyEngine(epsilon=1.0, delta=1e-5)
    assert engine.epsilon == 1.0
    assert engine.delta == 1e-5

@pytest.mark.privacy
def test_differential_privacy_guarantee():
    """Test that differential privacy guarantee is maintained."""
    # Implementation that verifies DP guarantee
    pass

@pytest.mark.slow
@pytest.mark.gpu
def test_training_with_privacy():
    """Test training with privacy (requires GPU)."""
    # Long-running test that requires GPU
    pass
```

### Running Tests

```bash
# All tests
make test

# Specific test types
make test-unit
make test-integration
make test-privacy

# With coverage
poetry run pytest --cov=privacy_finetuner

# Specific markers
poetry run pytest -m "unit and not slow"
```

### Test Data

- Use fixtures for common test data
- Mock external dependencies
- Never use real sensitive data in tests
- Create synthetic datasets for testing

## Documentation Guidelines

### Types of Documentation

1. **API Documentation**: Automatically generated from docstrings
2. **User Guides**: How-to guides for users
3. **Developer Documentation**: Architecture and implementation details
4. **Privacy Documentation**: Privacy mechanisms and guarantees

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add diagrams when helpful
- Update documentation with code changes

### Docstring Example

```python
def calculate_privacy_loss(
    epsilon: float,
    delta: float,
    steps: int
) -> float:
    """Calculate total privacy loss for a training run.
    
    Args:
        epsilon: Privacy parameter epsilon
        delta: Privacy parameter delta
        steps: Number of training steps
        
    Returns:
        Total privacy loss accumulated
        
    Raises:
        ValueError: If epsilon or delta are invalid
        
    Example:
        >>> loss = calculate_privacy_loss(1.0, 1e-5, 1000)
        >>> print(f"Privacy loss: {loss}")
    """
```

## Privacy and Security Guidelines

### Privacy Requirements

- **Never log sensitive data**
- **Implement differential privacy correctly**
- **Validate privacy parameters**
- **Monitor privacy budget consumption**
- **Test privacy guarantees**

### Security Requirements

- **Validate all inputs**
- **Use secure coding practices**
- **Handle secrets securely**
- **Implement proper authentication**
- **Follow least privilege principle**

### Privacy Code Review Checklist

- [ ] Privacy parameters are validated
- [ ] No sensitive data in logs
- [ ] Privacy budget is tracked
- [ ] Noise is added correctly
- [ ] Composition bounds are respected

## Code Review Process

### Submitting for Review

1. Ensure all tests pass
2. Run security and privacy checks
3. Update documentation
4. Fill out PR template completely
5. Request review from relevant team members

### Review Criteria

- **Functionality**: Does the code work as intended?
- **Privacy**: Are privacy guarantees maintained?
- **Security**: Are security best practices followed?
- **Performance**: Is the code efficient?
- **Maintainability**: Is the code clean and well-documented?
- **Testing**: Are there adequate tests?

### Addressing Feedback

- Respond to all comments
- Make requested changes promptly
- Test changes thoroughly
- Update documentation if needed
- Re-request review when ready

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Create release branch**
4. **Run full test suite**
5. **Create pull request**
6. **Merge and tag release**
7. **Publish to PyPI**
8. **Create GitHub release**

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community discussions
- **Discord**: Real-time chat (invite link in README)
- **Email**: [team@terragon-labs.com](mailto:team@terragon-labs.com)

### Getting Help

- Check existing issues and documentation first
- Use GitHub Discussions for questions
- Join our Discord for real-time help
- Attend office hours (announced in Discord)

### Recognition

We recognize contributors in several ways:
- Contributors listed in README
- Special recognition for significant contributions
- Invitation to contributor events
- Opportunity to present work at conferences

## Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `privacy`: Privacy-related issue
- `security`: Security-related issue
- `performance`: Performance improvement
- `dependencies`: Updates to dependencies

## Pull Request Template

When creating a pull request, use this template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Security fix
- [ ] Privacy enhancement

## Related Issues
Fixes #123

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Privacy tests pass
- [ ] Manual testing completed

## Privacy Review
- [ ] No sensitive data in logs
- [ ] Privacy parameters validated
- [ ] Privacy budget tracking correct
- [ ] Privacy guarantees maintained

## Security Review
- [ ] Input validation implemented
- [ ] No hardcoded secrets
- [ ] Authentication/authorization correct
- [ ] Security best practices followed

## Documentation
- [ ] Code comments added/updated
- [ ] API documentation updated
- [ ] User documentation updated
- [ ] CHANGELOG updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added for new functionality
- [ ] All checks pass
- [ ] Reviewers assigned
```

## Tips for Success

1. **Start small**: Begin with documentation fixes or small bug fixes
2. **Ask questions**: Don't hesitate to ask for help or clarification
3. **Be patient**: Code review takes time, especially for privacy-sensitive code
4. **Learn continuously**: Stay updated on privacy and security best practices
5. **Participate**: Engage with the community in discussions and reviews

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Privacy-Preserving Agent Finetuner! Your efforts help make privacy-preserving machine learning more accessible and secure for everyone.