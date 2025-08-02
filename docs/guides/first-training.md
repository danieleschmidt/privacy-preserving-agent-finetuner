# First Training Session

This guide walks you through running your first privacy-preserving fine-tuning session.

## Overview

You'll learn how to:
- Configure privacy settings
- Prepare a dataset
- Run a basic training session
- Understand privacy guarantees

## Prerequisites

- Privacy-Preserving Agent Fine-Tuner installed
- Basic understanding of machine learning
- Sample dataset (we'll provide one)

## Step 1: Basic Configuration

Create a basic privacy configuration:

```python
from privacy_finetuner import PrivateTrainer, PrivacyConfig

# Configure privacy budget
privacy_config = PrivacyConfig(
    epsilon=1.0,  # Privacy budget - lower = more private
    delta=1e-5,   # Privacy parameter
    max_grad_norm=1.0,  # Gradient clipping
    noise_multiplier=0.5
)
```

## Step 2: Initialize Trainer

```python
trainer = PrivateTrainer(
    model_name="microsoft/DialoGPT-small",  # Small model for demo
    privacy_config=privacy_config
)
```

## Step 3: Prepare Dataset

```python
# Use built-in sample dataset
trainer.load_sample_dataset("conversational")
```

## Step 4: Run Training

```python
# Run a short training session
results = trainer.train(
    epochs=1,
    batch_size=4,
    learning_rate=5e-5
)

print(f"Training completed with privacy budget ε={results.epsilon_spent}")
```

## Understanding Results

The training will output:
- **Privacy budget spent**: How much of your privacy budget was consumed
- **Model performance**: Accuracy/loss metrics
- **Privacy guarantees**: Formal differential privacy guarantees

## Next Steps

- [Privacy Budget Management](privacy-budgets.md) - Learn to optimize privacy budgets
- [Basic Configuration](basic-config.md) - Explore configuration options
- [Context Protection](context-protection.md) - Protect sensitive inputs