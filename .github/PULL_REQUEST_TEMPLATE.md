# Pull Request

## Description
Brief description of changes made in this PR.

## Type of Change
<!-- Mark the relevant option with an "x" -->
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔒 Security fix
- [ ] 🛡️ Privacy enhancement
- [ ] ⚡ Performance improvement
- [ ] 🧹 Code cleanup/refactoring
- [ ] 🔧 Build/CI changes

## Related Issues
<!-- Link to related issues -->
Fixes #(issue_number)
Related to #(issue_number)

## Changes Made
<!-- Detailed list of changes -->
- Change 1
- Change 2
- Change 3

## Testing
<!-- Describe the testing performed -->
- [ ] Unit tests pass (`make test-unit`)
- [ ] Integration tests pass (`make test-integration`)
- [ ] Privacy tests pass (`make test-privacy`)
- [ ] Security tests pass
- [ ] Manual testing completed
- [ ] Performance testing completed (if applicable)

### Test Evidence
<!-- Provide evidence of testing -->
```bash
# Example test output
$ make test
...
PASSED
```

## Privacy Review
<!-- Complete privacy impact assessment -->
- [ ] No sensitive data in logs or outputs
- [ ] Privacy parameters validated and documented
- [ ] Privacy budget tracking implemented correctly
- [ ] Differential privacy guarantees maintained
- [ ] Context protection mechanisms work as expected
- [ ] Privacy compliance checks pass

### Privacy Impact
<!-- Describe privacy implications -->
- **Data processed**: [describe what data is processed]
- **Privacy mechanisms**: [list privacy mechanisms used]
- **Privacy parameters**: [document epsilon, delta, etc.]
- **Compliance**: [GDPR/HIPAA/CCPA considerations]

## Security Review
<!-- Complete security assessment -->
- [ ] Input validation implemented
- [ ] No hardcoded secrets or credentials
- [ ] Authentication/authorization correct
- [ ] Security best practices followed
- [ ] No new attack vectors introduced
- [ ] Security scans pass

### Security Checklist
- [ ] SQL injection prevention
- [ ] XSS prevention
- [ ] CSRF protection
- [ ] Secure data transmission
- [ ] Proper error handling
- [ ] Access control validation

## Performance Impact
<!-- Assess performance implications -->
- [ ] No significant performance regression
- [ ] Memory usage is acceptable
- [ ] CPU usage is acceptable
- [ ] GPU usage is acceptable (if applicable)
- [ ] Benchmarks completed (if applicable)

### Performance Metrics
<!-- Include relevant metrics -->
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Memory | X MB   | Y MB  | +Z MB  |
| CPU    | X%     | Y%    | +Z%    |
| Time   | X sec  | Y sec | +Z sec |

## Documentation
<!-- Documentation updates -->
- [ ] Code comments added/updated
- [ ] API documentation updated
- [ ] User documentation updated
- [ ] Architecture documentation updated
- [ ] CHANGELOG.md updated
- [ ] README updated (if needed)

## Deployment
<!-- Deployment considerations -->
- [ ] Database migrations included (if applicable)
- [ ] Configuration changes documented
- [ ] Environment variables updated
- [ ] Docker image builds successfully
- [ ] Deployment runbook updated (if needed)

## Breaking Changes
<!-- If this is a breaking change, describe the impact and migration path -->
### Impact
- What breaks:
- Who is affected:

### Migration Path
- Steps to migrate:
- Timeline:
- Support provided:

## Dependencies
<!-- New or updated dependencies -->
- [ ] No new dependencies added
- [ ] New dependencies are justified and documented
- [ ] Dependency licenses are compatible
- [ ] Security scan of dependencies completed

### New Dependencies
| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| example | 1.0.0   | Purpose | MIT     |

## Checklist
<!-- Final checklist before review -->
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added for new functionality
- [ ] All checks pass in CI/CD
- [ ] Privacy compliance verified
- [ ] Security review completed
- [ ] Documentation updated
- [ ] Breaking changes documented
- [ ] Reviewers assigned

## Reviewer Guidelines
<!-- Guidelines for reviewers -->
### Focus Areas
- [ ] Code quality and maintainability
- [ ] Privacy protection mechanisms
- [ ] Security considerations
- [ ] Performance impact
- [ ] Test coverage and quality
- [ ] Documentation completeness

### Privacy Review Checklist
- [ ] Verify privacy parameters are reasonable
- [ ] Check that privacy budget is tracked correctly
- [ ] Ensure no sensitive data leakage
- [ ] Validate differential privacy implementation
- [ ] Review context protection mechanisms

### Security Review Checklist
- [ ] Validate input sanitization
- [ ] Check authentication/authorization
- [ ] Review error handling
- [ ] Verify secure communication
- [ ] Check for common vulnerabilities

## Additional Notes
<!-- Any additional information for reviewers -->

## Screenshots/Demos
<!-- Include relevant screenshots or demo links -->

---

**By submitting this pull request, I confirm that:**
- [ ] My code follows the project's code of conduct
- [ ] I have read and followed the contributing guidelines
- [ ] My changes do not introduce any privacy or security vulnerabilities
- [ ] I have tested my changes thoroughly
- [ ] I grant permission for my contribution to be licensed under the project's license