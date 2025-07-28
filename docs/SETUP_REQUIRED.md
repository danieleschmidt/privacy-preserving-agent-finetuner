# Manual Setup Requirements

## GitHub Repository Configuration

### Branch Protection Rules
1. Navigate to Settings → Branches
2. Add rule for `main` branch:
   - Require pull request reviews (min 2)
   - Require status checks to pass
   - Require branches to be up to date
   - Restrict pushes to matching branches

### Repository Settings
1. **General Settings**:
   - Enable issues, wikis, discussions
   - Set visibility to appropriate level
   - Configure merge options (squash merge recommended)

2. **Security & Analysis**:
   - Enable dependency graph
   - Enable Dependabot alerts and security updates
   - Enable secret scanning
   - Enable code scanning (CodeQL)

### Secrets Configuration
Add these repository secrets in Settings → Secrets and variables:
- `PYPI_API_TOKEN` - For package publishing
- `CODECOV_TOKEN` - For test coverage reporting
- `SNYK_TOKEN` - For security scanning
- `DOCKER_HUB_TOKEN` - For container publishing

## GitHub Actions Workflows

Copy example workflows from `docs/workflows/examples/` to `.github/workflows/`:

```bash
cp docs/workflows/examples/*.yml .github/workflows/
```

### Required Permissions
Ensure workflows have appropriate permissions in repository settings:
- Actions: Read and write permissions
- Contents: Write (for releases)
- Issues: Write (for automated issue creation)
- Pull requests: Write (for status updates)

## External Service Integration

### 1. Code Coverage (Codecov)
- Register repository at [codecov.io](https://codecov.io)
- Add `CODECOV_TOKEN` to repository secrets
- Configure coverage thresholds in `codecov.yml`

### 2. Security Scanning (Snyk)
- Register at [snyk.io](https://snyk.io) 
- Connect GitHub repository
- Add `SNYK_TOKEN` to repository secrets
- Configure vulnerability policies

### 3. Container Registry
- Set up Docker Hub or GitHub Container Registry
- Add authentication tokens to secrets
- Configure automated builds

### 4. Documentation Hosting
- Set up ReadTheDocs or GitHub Pages
- Configure automatic builds from main branch
- Link documentation in repository settings

## Monitoring & Observability

### Prometheus/Grafana Setup
1. Deploy monitoring stack (see `monitoring/` directory)
2. Configure service discovery for application metrics
3. Import Grafana dashboard from `monitoring/grafana/dashboards/`
4. Set up alerting rules from `monitoring/alert_rules.yml`

### Log Aggregation
1. Configure centralized logging (ELK/EFK stack)
2. Set up log forwarding from application
3. Create log-based alerts for security events
4. Configure retention policies

## Compliance & Governance

### Privacy Impact Assessment
1. Complete privacy impact assessment template
2. Document data flows and processing activities
3. Validate differential privacy implementations
4. Regular compliance audits (quarterly)

### Security Audit
1. Perform penetration testing (annually)
2. Code security reviews for major releases  
3. Dependency vulnerability assessments
4. Access control reviews (quarterly)

## Package Publishing

### PyPI Configuration
1. Create PyPI account and API token
2. Add token to repository secrets
3. Test publishing to TestPyPI first
4. Configure trusted publishing with GitHub Actions

### Release Process
1. Update version in `pyproject.toml`
2. Create release notes in `CHANGELOG.md`
3. Create and push version tag
4. Monitor automated release workflow

## Team Permissions

Configure team access levels:
- **Maintainers**: Admin access to repository
- **Contributors**: Write access for core team
- **Community**: Triage access for active contributors
- **External**: Read access with PR submission ability

## Checklist

- [ ] Branch protection rules configured
- [ ] Repository secrets added
- [ ] Workflow files copied and customized
- [ ] External services integrated
- [ ] Monitoring stack deployed
- [ ] Documentation site configured
- [ ] Package publishing tested
- [ ] Team permissions set
- [ ] Compliance documentation completed