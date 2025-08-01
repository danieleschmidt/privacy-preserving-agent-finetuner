# ADR-001: Differential Privacy Framework Choice

## Status

Accepted

## Context

The privacy-preserving agent fine-tuner requires robust differential privacy guarantees to ensure sensitive data protection during model training. Multiple frameworks are available including Opacus, TensorFlow Privacy, and custom implementations.

## Decision

We will use **Opacus** as the primary differential privacy framework with PyTorch integration, supplemented by custom privacy accounting for advanced scenarios.

## Consequences

### Positive Consequences

- Strong theoretical foundations with formal privacy guarantees (ε-δ differential privacy)
- Excellent PyTorch integration for seamless model training
- Active development and community support from Meta AI
- Comprehensive privacy accounting with RDP (Rényi Differential Privacy)
- Support for both individual and batch-level privacy
- Flexible privacy budget management

### Negative Consequences

- PyTorch dependency limits framework flexibility
- Additional computational overhead during training (15-20%)
- Memory usage increase for gradient clipping and noise addition
- Learning curve for privacy parameter tuning

### Neutral Consequences

- Need for custom extensions for federated learning scenarios
- Additional validation required for privacy budget calculations

## Implementation Notes

- Configure Opacus PrivacyEngine with careful attention to:
  - Gradient clipping norms (typically 1.0-2.0)
  - Noise multipliers (0.5-1.5 range)
  - Privacy accounting methods (RDP preferred)
- Implement custom privacy auditing for compliance requirements
- Add privacy budget monitoring and alerting

## Alternatives Considered

1. **TensorFlow Privacy**: Rejected due to TensorFlow dependency and less mature PyTorch ecosystem integration
2. **Custom Implementation**: Rejected due to complexity of privacy accounting and higher risk of implementation errors
3. **PySyft**: Considered for federated learning but Opacus provides better single-node privacy guarantees

## References

- [Opacus Documentation](https://opacus.ai/)
- [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- [Meta AI Opacus Paper](https://arxiv.org/abs/2109.12298)