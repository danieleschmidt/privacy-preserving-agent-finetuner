#!/usr/bin/env python3
"""
Basic Privacy-Preserving Training Example

This example demonstrates how to use the Privacy-Preserving Agent Finetuner
for basic differential privacy training with a small dataset.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from privacy_finetuner.core import PrivateTrainer, PrivacyConfig, ContextGuard, RedactionStrategy
from privacy_finetuner.utils.logging_config import setup_privacy_logging

def create_sample_dataset(output_path: str):
    """Create a small sample dataset for training."""
    sample_data = [
        {"text": "The weather is beautiful today."},
        {"text": "Machine learning helps automate complex tasks."},
        {"text": "Privacy protection is essential in AI systems."},
        {"text": "Differential privacy provides mathematical guarantees."},
        {"text": "Natural language models can understand context."},
        {"text": "Training on sensitive data requires special care."},
        {"text": "Federated learning enables distributed training."},
        {"text": "Encryption protects data during transmission."},
        {"text": "Secure computation allows private inference."},
        {"text": "Compliance with regulations is mandatory."}
    ]
    
    import json
    with open(output_path, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✓ Created sample dataset with {len(sample_data)} examples: {output_path}")

def basic_training_example():
    """Demonstrate basic privacy-preserving training."""
    
    # Setup logging
    setup_privacy_logging(
        log_level="INFO",
        log_file="logs/training.log",
        structured_logging=True,
        privacy_redaction=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting basic privacy-preserving training example")
    
    try:
        # Create sample dataset
        dataset_path = "sample_dataset.jsonl"
        create_sample_dataset(dataset_path)
        
        # Configure privacy settings
        privacy_config = PrivacyConfig(
            epsilon=1.0,              # Privacy budget
            delta=1e-5,               # Failure probability
            max_grad_norm=1.0,        # Gradient clipping
            noise_multiplier=0.8,     # Noise level
            accounting_mode="rdp"     # RDP accounting
        )
        
        logger.info(f"Privacy configuration: ε={privacy_config.epsilon}, δ={privacy_config.delta}")
        
        # Validate privacy configuration
        privacy_config.validate()
        logger.info("✓ Privacy configuration validated")
        
        # Initialize trainer (this will warn about missing dependencies in demo environment)
        try:
            trainer = PrivateTrainer(
                model_name="microsoft/DialoGPT-small",  # Small model for testing
                privacy_config=privacy_config,
                use_mcp_gateway=False  # Disable for simplicity
            )
            
            logger.info("✓ Trainer initialized successfully")
            
            # Demonstrate training (will fail gracefully without ML libraries)
            try:
                results = trainer.train(
                    dataset=dataset_path,
                    epochs=1,
                    batch_size=2,
                    learning_rate=5e-5
                )
                
                logger.info("✓ Training completed successfully")
                logger.info(f"Privacy spent: ε={results.get('privacy_spent', 0):.6f}")
                
                # Generate privacy report
                privacy_report = trainer.get_privacy_report()
                logger.info("Privacy Report:")
                logger.info(f"  Epsilon spent: {privacy_report['epsilon_spent']:.6f}")
                logger.info(f"  Remaining budget: {privacy_report['remaining_budget']:.6f}")
                
            except Exception as e:
                logger.info(f"Training demo failed (expected in environment without ML libraries): {type(e).__name__}")
                logger.info("This is expected - full training requires PyTorch, Transformers, and other ML dependencies")
                
        except Exception as e:
            logger.info(f"Trainer initialization failed (expected): {type(e).__name__}")
            logger.info("This demonstrates graceful handling of missing dependencies")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        return 1
    
    logger.info("🎉 Basic training example completed successfully")
    return 0

def context_protection_example():
    """Demonstrate context protection functionality."""
    
    logger = logging.getLogger(__name__)
    logger.info("🛡️ Starting context protection example")
    
    # Sample sensitive text
    sensitive_texts = [
        "Contact John Doe at john.doe@company.com or call 555-123-4567",
        "Process payment for credit card 4111-1111-1111-1111",
        "Patient SSN is 123-45-6789 and lives in New York",
        "API key: sk-1234567890abcdef, use with caution",
        "Medical record shows patient has diabetes and hypertension"
    ]
    
    # Initialize context guard
    guard = ContextGuard([
        RedactionStrategy.PII_REMOVAL,
        RedactionStrategy.ENTITY_HASHING
    ])
    
    logger.info("✓ Context guard initialized")
    
    for i, text in enumerate(sensitive_texts, 1):
        logger.info(f"\n--- Example {i} ---")
        logger.info(f"Original: {text}")
        
        # Analyze sensitivity
        analysis = guard.analyze_sensitivity(text)
        logger.info(f"Sensitivity: {analysis['sensitivity_level']} (score: {analysis['sensitivity_score']})")
        
        # Apply protection
        protected = guard.protect(text, analysis['sensitivity_level'])
        logger.info(f"Protected: {protected}")
        
        # Generate report
        report = guard.create_privacy_report(text, protected)
        logger.info(f"Compliance: GDPR={report['privacy_compliance']['gdpr_compliant']}")
    
    logger.info("✓ Context protection examples completed")

def main():
    """Run all examples."""
    
    # Create directories
    Path("logs").mkdir(exist_ok=True)
    
    print("Privacy-Preserving Agent Finetuner - Basic Examples")
    print("=" * 60)
    
    # Run examples
    try:
        print("\n1. Context Protection Example")
        context_protection_example()
        
        print("\n2. Basic Training Example")
        basic_training_example()
        
        print("\n✅ All examples completed successfully!")
        print("\nNext steps:")
        print("- Install ML dependencies: pip install torch transformers datasets opacus")
        print("- Try with real datasets and models")
        print("- Explore advanced privacy configurations")
        print("- Set up monitoring and compliance dashboards")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())