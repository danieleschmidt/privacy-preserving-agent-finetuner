# GitHub Actions Workflow Setup Guide

## Overview

This guide provides comprehensive GitHub Actions workflow templates for the Privacy-Preserving Agent Finetuner project. These workflows implement enterprise-grade CI/CD with privacy-specific testing and compliance verification.

## Required Workflow Files

Create these workflow files in `.github/workflows/` directory:

### 1. Core CI/CD Workflow (`ci.yml`)

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: "3.11"
  POETRY_VERSION: "1.7.1"

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        version: ${{ env.POETRY_VERSION }}
    
    - name: Configure Poetry
      run: |
        poetry config virtualenvs.create true
        poetry config virtualenvs.in-project true
    
    - name: Cache dependencies
      uses: actions/cache@v4
      with:
        path: .venv
        key: venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('**/poetry.lock') }}
    
    - name: Install dependencies
      run: poetry install --with dev
    
    - name: Run privacy compliance checks
      run: poetry run python scripts/privacy_compliance_check.py
    
    - name: Run security audit
      run: poetry run python scripts/security_audit.py
    
    - name: Run tests with privacy markers
      run: |
        poetry run pytest tests/ -v \
          --cov=privacy_finetuner \
          --cov-report=xml \
          --cov-report=html \
          -m "not slow"
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v4
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  privacy-tests:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
    
    - name: Install dependencies
      run: poetry install --with dev
    
    - name: Run privacy guarantee tests
      run: |
        poetry run pytest tests/test_privacy_guarantees.py -v \
          --privacy-budget=1.0 \
          -m "privacy"
    
    - name: Generate privacy report
      run: poetry run python scripts/generate_privacy_report.py
    
    - name: Upload privacy artifacts
      uses: actions/upload-artifact@v4
      with:
        name: privacy-reports
        path: reports/privacy/

  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Generate SBOM
      run: |
        pip install cyclonedx-bom
        cyclonedx-py -o sbom.json
    
    - name: Upload SBOM
      uses: actions/upload-artifact@v4
      with:
        name: sbom
        path: sbom.json

  docker-build:
    runs-on: ubuntu-latest
    needs: [test, privacy-tests, security-scan]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: false
        tags: privacy-finetuner:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Run container security scan
      run: |
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
          aquasec/trivy image privacy-finetuner:${{ github.sha }}
```

### 2. Release Workflow (`release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

env:
  PYTHON_VERSION: "3.11"

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
    
    - name: Build package
      run: poetry build
    
    - name: Generate SBOM for release
      run: poetry run python scripts/generate_sbom.py --output sbom-release.json
    
    - name: Generate privacy compliance report
      run: poetry run python scripts/privacy_compliance_check.py --output compliance-report.json
    
    - name: Create Release
      uses: softprops/action-gh-release@v1
      with:
        files: |
          dist/*
          sbom-release.json
          compliance-report.json
        generate_release_notes: true
        draft: false
        prerelease: ${{ contains(github.ref, 'alpha') || contains(github.ref, 'beta') || contains(github.ref, 'rc') }}
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Publish to PyPI
      env:
        POETRY_PYPI_TOKEN_PYPI: ${{ secrets.PYPI_TOKEN }}
      run: poetry publish
```

### 3. Security Monitoring (`security.yml`)

```yaml
name: Security Monitoring

on:
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday
  workflow_dispatch:

jobs:
  security-audit:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
    
    - name: Install dependencies
      run: poetry install --with dev
    
    - name: Run comprehensive security audit
      run: poetry run python scripts/advanced_security_scanner.py --output security-audit.json
    
    - name: Check for dependency vulnerabilities
      run: poetry run safety check --json > safety-report.json
    
    - name: Generate updated SBOM
      run: poetry run python scripts/generate_sbom.py --output sbom-current.json
    
    - name: Upload security artifacts
      uses: actions/upload-artifact@v4
      with:
        name: security-reports-${{ github.run_number }}
        path: |
          security-audit.json
          safety-report.json
          sbom-current.json
    
    - name: Create security issue if vulnerabilities found
      uses: actions/github-script@v7
      with:
        script: |
          const fs = require('fs');
          try {
            const safetyReport = JSON.parse(fs.readFileSync('safety-report.json', 'utf8'));
            if (safetyReport.vulnerabilities && safetyReport.vulnerabilities.length > 0) {
              github.rest.issues.create({
                owner: context.repo.owner,
                repo: context.repo.repo,
                title: '🚨 Security Vulnerabilities Detected',
                body: `Automated security scan detected ${safetyReport.vulnerabilities.length} vulnerabilities. Please review the attached reports.`,
                labels: ['security', 'high-priority']
              });
            }
          } catch (error) {
            console.log('No vulnerabilities file found or parsing error:', error.message);
          }
```

### 4. Performance Monitoring (`performance.yml`)

```yaml
name: Performance Monitoring

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 12 * * *'  # Daily at noon

jobs:
  performance-benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
    
    - name: Install dependencies
      run: poetry install --with dev
    
    - name: Run performance profiling
      run: |
        poetry run python scripts/performance_profiler.py \
          --output performance-report.json \
          --include-privacy-overhead
    
    - name: Upload performance data
      uses: actions/upload-artifact@v4
      with:
        name: performance-${{ github.sha }}
        path: performance-report.json
    
    - name: Comment performance results on PR
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v7
      with:
        script: |
          const fs = require('fs');
          try {
            const report = JSON.parse(fs.readFileSync('performance-report.json', 'utf8'));
            const comment = `## 📊 Performance Report
            
            **Privacy Training Overhead**: ${report.privacy_overhead_percent}%
            **Memory Usage**: ${report.peak_memory_mb} MB
            **Training Speed**: ${report.tokens_per_second} tokens/sec
            
            ${report.performance_regression ? '⚠️ Performance regression detected' : '✅ Performance within acceptable limits'}`;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
          } catch (error) {
            console.log('Error reading performance report:', error.message);
          }
```

## Secrets Configuration

Configure these secrets in your GitHub repository settings:

### Required Secrets
- `CODECOV_TOKEN`: For code coverage reporting
- `PYPI_TOKEN`: For publishing to PyPI
- `DOCKER_HUB_USERNAME`: For Docker image publishing
- `DOCKER_HUB_ACCESS_TOKEN`: For Docker Hub authentication

### Optional Secrets
- `SLACK_WEBHOOK`: For Slack notifications
- `DATADOG_API_KEY`: For monitoring integration
- `SECURITY_SCAN_TOKEN`: For additional security scanning services

## Branch Protection Rules

Configure these branch protection rules for `main` branch:

```yaml
protection_rules:
  required_status_checks:
    strict: true
    contexts:
      - "test (3.9)"
      - "test (3.10)" 
      - "test (3.11)"
      - "test (3.12)"
      - "privacy-tests"
      - "security-scan"
      - "docker-build"
  
  enforce_admins: true
  required_pull_request_reviews:
    required_approving_review_count: 2
    dismiss_stale_reviews: true
    require_code_owner_reviews: true
  
  restrictions: null
  allow_force_pushes: false
  allow_deletions: false
```

## Integration with External Services

### CodeCov Integration
1. Enable CodeCov for your repository
2. Add `CODECOV_TOKEN` to secrets
3. Coverage reports will be automatically uploaded

### Security Scanning Integration
- **Trivy**: Vulnerability scanning for containers and filesystems
- **Safety**: Python dependency vulnerability scanning
- **Bandit**: Static security analysis for Python code
- **detect-secrets**: Secret detection in code

### Performance Monitoring
- Automated performance regression detection
- Privacy overhead measurement
- Memory and GPU utilization tracking
- Benchmark comparison across commits

## Compliance Features

### Privacy Compliance
- Automated privacy guarantee verification
- GDPR compliance checking
- HIPAA compliance validation
- EU AI Act compliance verification

### Regulatory Reporting
- Automated SBOM generation for supply chain security
- Privacy impact assessments
- Security audit trails
- Compliance certification reports

## Customization

### Environment-Specific Configuration
Create environment-specific workflow files:
- `.github/workflows/staging.yml`
- `.github/workflows/production.yml`
- `.github/workflows/development.yml`

### Custom Test Environments
Configure matrix builds for different environments:
```yaml
strategy:
  matrix:
    include:
      - os: ubuntu-latest
        python-version: "3.11"
        env: "cpu"
      - os: ubuntu-latest
        python-version: "3.11"
        env: "gpu"
        runs-on: [self-hosted, gpu]
```

## Monitoring and Alerting

### GitHub Actions Monitoring
- Workflow execution time tracking
- Failure rate monitoring
- Resource usage optimization
- Cost optimization recommendations

### Integration Points
- Slack notifications for failures
- Email alerts for security issues
- Dashboard integration for metrics
- PagerDuty for critical failures

## Troubleshooting

### Common Issues
1. **Poetry Lock File Issues**: Ensure `poetry.lock` is committed
2. **Secret Access**: Verify all required secrets are configured
3. **Dependency Conflicts**: Check for version compatibility
4. **Resource Limits**: Monitor workflow execution times

### Debug Mode
Enable debug logging by setting `ACTIONS_STEP_DEBUG: true` in workflow environment variables.

---

**Note**: Remember to adapt these templates to your specific repository requirements and security policies. Always test workflows in a development environment before deploying to production.