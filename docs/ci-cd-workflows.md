# CI/CD Workflows Documentation

## Overview

This document describes the CI/CD workflows that should be implemented for the Privacy-Preserving Agent Finetuner project. Since GitHub workflows cannot be automatically created, this serves as a comprehensive guide for manual implementation.

## Required GitHub Workflows

### 1. Pull Request Validation (`.github/workflows/pr-validation.yml`)

```yaml
name: Pull Request Validation

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main, develop]

jobs:
  code-quality:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]
    
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install Poetry
        uses: snok/install-poetry@v1
        with:
          version: 1.7.1
          virtualenvs-create: true
          virtualenvs-in-project: true
      
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: .venv
          key: venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('poetry.lock') }}
      
      - name: Install dependencies
        run: poetry install --with dev
      
      - name: Run linting
        run: |
          poetry run ruff check .
          poetry run black --check .
          poetry run isort --check-only .
          poetry run mypy privacy_finetuner/
      
      - name: Run tests
        run: |
          poetry run pytest tests/ -v --cov=privacy_finetuner --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: python
      
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2
      
      - name: Run Bandit
        run: |
          pip install bandit
          bandit -r privacy_finetuner/ -f json -o bandit-report.json
      
      - name: Run Safety
        run: |
          pip install safety
          safety check --json --output safety-report.json
      
      - name: Run Trivy
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: |
          docker build -t privacy-finetuner:test .
          docker run --rm privacy-finetuner:test python -c "import privacy_finetuner; print('Import successful')"
```

### 2. Release Workflow (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.11
      
      - name: Install Poetry
        uses: snok/install-poetry@v1
      
      - name: Build package
        run: poetry build
      
      - name: Publish to PyPI
        env:
          POETRY_PYPI_TOKEN_PYPI: ${{ secrets.PYPI_TOKEN }}
        run: poetry publish
      
      - name: Build and push Docker image
        env:
          REGISTRY: ghcr.io
          IMAGE_NAME: terragon-labs/privacy-finetuner
        run: |
          echo ${{ secrets.GITHUB_TOKEN }} | docker login $REGISTRY -u ${{ github.actor }} --password-stdin
          docker build -t $REGISTRY/$IMAGE_NAME:${{ github.ref_name }} .
          docker build -t $REGISTRY/$IMAGE_NAME:latest .
          docker push $REGISTRY/$IMAGE_NAME:${{ github.ref_name }}
          docker push $REGISTRY/$IMAGE_NAME:latest
      
      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          draft: false
          prerelease: false
```

### 3. Deployment Workflow (`.github/workflows/deploy.yml`)

```yaml
name: Deploy

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy to'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

jobs:
  deploy-staging:
    if: github.ref == 'refs/heads/main' || github.event.inputs.environment == 'staging'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to staging
        run: |
          echo "Deploying to staging environment"
          # Add actual deployment commands here
      
      - name: Run smoke tests
        run: |
          echo "Running smoke tests"
          # Add smoke test commands here

  deploy-production:
    if: github.event.inputs.environment == 'production'
    runs-on: ubuntu-latest
    environment: production
    needs: [deploy-staging]
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to production
        run: |
          echo "Deploying to production environment"
          # Add actual deployment commands here
      
      - name: Run health checks
        run: |
          echo "Running health checks"
          # Add health check commands here
```

### 4. Security Monitoring (`.github/workflows/security-monitoring.yml`)

```yaml
name: Security Monitoring

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  dependency-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run dependency check
        uses: dependency-check/Dependency-Check_Action@main
        with:
          project: 'privacy-finetuner'
          path: '.'
          format: 'ALL'
          out: 'reports'
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: dependency-check-report
          path: reports/

  license-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: License check
        uses: fossa-contrib/fossa-action@v2
        with:
          api-key: ${{ secrets.FOSSA_API_KEY }}
```

## Branch Protection Rules

Configure the following branch protection rules for the `main` branch:

1. **Required status checks**:
   - `code-quality`
   - `security-scan`
   - `build-test`

2. **Restrictions**:
   - Require pull request reviews before merging
   - Dismiss stale reviews when new commits are pushed
   - Require review from code owners
   - Restrict pushes to matching branches

3. **Rules**:
   - Require branches to be up to date before merging
   - Require signed commits
   - Include administrators in restrictions

## Environment Configuration

### Staging Environment
- **URL**: `https://staging.privacy-finetuner.terragon-labs.com`
- **Required reviewers**: 1
- **Deployment protection rules**: None
- **Environment secrets**:
  - `DATABASE_URL`
  - `REDIS_URL`
  - `PRIVACY_CONFIG`

### Production Environment
- **URL**: `https://privacy-finetuner.terragon-labs.com`
- **Required reviewers**: 2
- **Deployment protection rules**: Wait timer (10 minutes)
- **Environment secrets**:
  - `DATABASE_URL`
  - `REDIS_URL`
  - `PRIVACY_CONFIG`
  - `MONITORING_API_KEY`

## Required Secrets

Add the following secrets to your GitHub repository:

### General Secrets
- `PYPI_TOKEN`: PyPI API token for package publishing
- `DOCKER_REGISTRY_TOKEN`: Container registry authentication
- `CODECOV_TOKEN`: Code coverage reporting

### Security Secrets
- `FOSSA_API_KEY`: License compliance scanning
- `SNYK_TOKEN`: Vulnerability scanning
- `SONARCLOUD_TOKEN`: Code quality analysis

### Deployment Secrets
- `AWS_ACCESS_KEY_ID`: AWS deployment credentials
- `AWS_SECRET_ACCESS_KEY`: AWS deployment credentials
- `KUBE_CONFIG`: Kubernetes configuration for deployment

### Monitoring Secrets
- `SENTRY_DSN`: Error tracking
- `DATADOG_API_KEY`: Performance monitoring
- `PROMETHEUS_CONFIG`: Metrics collection

## Deployment Strategy

### Blue-Green Deployment
1. Deploy new version to green environment
2. Run health checks and smoke tests
3. Switch traffic from blue to green
4. Keep blue environment as rollback option

### Canary Deployment
1. Deploy to 5% of production traffic
2. Monitor metrics and error rates
3. Gradually increase traffic (10%, 25%, 50%, 100%)
4. Rollback if issues detected

## Monitoring and Alerting

### Key Metrics to Monitor
- **Performance**: Response time, throughput, error rate
- **Privacy**: Budget consumption, noise levels
- **Security**: Failed authentication, anomalous access
- **Infrastructure**: CPU, memory, disk usage

### Alert Thresholds
- **Critical**: Error rate > 5%, Response time > 5s
- **Warning**: Error rate > 1%, Response time > 2s
- **Info**: New deployment, configuration change

## Rollback Procedures

### Automatic Rollback Triggers
- Error rate exceeds 5% for 5 minutes
- Response time exceeds 10s for 3 minutes
- Health check failures for 2 minutes

### Manual Rollback Steps
1. Identify problematic deployment
2. Switch traffic to previous version
3. Investigate and fix issues
4. Prepare hotfix if necessary
5. Re-deploy with fixes

## Compliance and Auditing

### Audit Requirements
- All deployments must be logged
- Code changes require peer review
- Security scans must pass
- Privacy compliance checks must pass

### Compliance Reports
- Weekly security scan summary
- Monthly privacy audit report
- Quarterly compliance review
- Annual penetration testing

## Best Practices

1. **Never deploy on Fridays** (unless critical security fix)
2. **Always run tests in staging** before production
3. **Monitor deployments actively** for first 30 minutes
4. **Keep rollback scripts updated** and tested
5. **Document all incidents** and lessons learned
6. **Review and update workflows quarterly**
7. **Train team members** on deployment procedures
8. **Maintain deployment runbooks** for common scenarios

## Implementation Checklist

- [ ] Create GitHub workflows in `.github/workflows/`
- [ ] Configure branch protection rules
- [ ] Set up staging and production environments
- [ ] Add required secrets to repository
- [ ] Configure monitoring and alerting
- [ ] Test deployment workflows
- [ ] Document incident response procedures
- [ ] Train team on CI/CD processes
- [ ] Schedule regular reviews and updates