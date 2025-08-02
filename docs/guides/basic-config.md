# Basic Configuration

Learn how to configure the Privacy-Preserving Agent Fine-Tuner for your specific needs.

## Configuration Overview

The system uses YAML configuration files for most settings. The main configuration areas are:

- **Privacy settings**: Differential privacy parameters
- **Model settings**: Base model and training parameters  
- **Context protection**: Input sanitization strategies
- **Monitoring**: Privacy budget and performance tracking

## Privacy Configuration

Create `config/privacy.yaml`:

```yaml
privacy:
  epsilon: 1.0              # Privacy budget (lower = more private)
  delta: 1e-5              # Privacy parameter
  max_grad_norm: 1.0       # Gradient clipping threshold
  noise_multiplier: 0.5    # Noise scale for differential privacy
  
  # Advanced settings
  accounting_mode: "rdp"    # Privacy accounting method
  target_delta: 1e-5       # Target delta for composition
```

## Model Configuration

Create `config/models.yaml`:

```yaml
models:
  base_model: "meta-llama/Llama-2-7b-hf"
  
  training:
    batch_size: 8
    gradient_accumulation_steps: 4
    learning_rate: 5e-5
    warmup_steps: 100
    max_steps: 1000
    
  adapters:
    type: "lora"            # Low-rank adaptation
    r: 8                    # Rank parameter
    alpha: 16              # Scaling parameter
    dropout: 0.1           # Dropout rate
```

## Context Protection

Configure input sanitization in `config/privacy.yaml`:

```yaml
context_protection:
  strategies:
    - type: "pii_removal"
      sensitivity: "high"
    - type: "entity_hashing"  
      salt: "${HASH_SALT}"
    - type: "semantic_encryption"
      key_rotation_hours: 24
```

## Environment Variables

Set these environment variables:

```bash
export PRIVACY_CONFIG_PATH="./config/privacy.yaml"
export MODEL_CONFIG_PATH="./config/models.yaml"
export HASH_SALT="your-random-salt-here"
export CUDA_VISIBLE_DEVICES="0"  # GPU selection
```

## Validation

Validate your configuration:

```python
from privacy_finetuner import validate_config

# Validate configuration files
is_valid, errors = validate_config("config/")
if not is_valid:
    print("Configuration errors:", errors)
```

## Common Configurations

### High Privacy (Healthcare/Finance)
```yaml
privacy:
  epsilon: 0.1      # Very low privacy budget
  delta: 1e-6      # Very small delta
  max_grad_norm: 0.5
  noise_multiplier: 2.0
```

### Balanced Privacy (General Use)
```yaml
privacy:
  epsilon: 1.0      # Moderate privacy budget
  delta: 1e-5      # Standard delta
  max_grad_norm: 1.0
  noise_multiplier: 0.5
```

### Performance Focus (Less Sensitive Data)
```yaml
privacy:
  epsilon: 10.0     # Higher privacy budget
  delta: 1e-4      # Larger delta
  max_grad_norm: 2.0
  noise_multiplier: 0.1
```

## Next Steps

- [Privacy Budget Management](privacy-budgets.md) - Optimize privacy budgets
- [Differential Privacy Explained](differential-privacy.md) - Understand the theory
- [Performance Optimization](user/performance-optimization.md) - Improve training speed