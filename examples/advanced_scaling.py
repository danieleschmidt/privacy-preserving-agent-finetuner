#!/usr/bin/env python3
"""
Advanced Scaling and Optimization Example

This example demonstrates advanced scaling features including:
- Resource optimization and auto-configuration
- Federated learning with multiple clients
- Performance monitoring and recommendations
- Distributed training capabilities
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from privacy_finetuner.core import PrivacyConfig
from privacy_finetuner.optimization.resource_optimizer import ResourceOptimizer, ResourceProfile
from privacy_finetuner.distributed.federated_trainer import FederatedPrivateTrainer, FederatedConfig, AggregationMethod
from privacy_finetuner.utils.logging_config import setup_privacy_logging, performance_monitor

def resource_optimization_example():
    """Demonstrate resource optimization and auto-configuration."""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting resource optimization example")
    
    # Detect system resources
    resource_profile = ResourceProfile.detect_system_profile()
    logger.info(f"Detected system: {resource_profile.cpu_cores} CPU cores, "
               f"{resource_profile.gpu_count} GPUs, {resource_profile.total_memory}GB RAM")
    
    # Initialize resource optimizer
    optimizer = ResourceOptimizer(
        resource_profile=resource_profile,
        optimization_target="throughput"
    )
    
    # Define a hypothetical training scenario
    model_size = 125_000_000  # 125M parameters (BERT-like)
    dataset_size = 100_000    # 100K samples
    target_privacy_budget = 1.0
    time_constraint = 12.0    # 12 hours
    
    logger.info(f"Optimizing for: {model_size:,} parameter model, "
               f"{dataset_size:,} samples, ε={target_privacy_budget}")
    
    # Get optimized configuration
    optimized_config = optimizer.optimize_training_configuration(
        model_size=model_size,
        dataset_size=dataset_size, 
        target_privacy_budget=target_privacy_budget,
        time_constraint=time_constraint
    )
    
    logger.info("📊 Optimized Training Configuration:")
    for key, value in optimized_config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Monitor current resource usage
    logger.info("\n📈 Current Resource Usage:")
    resource_usage = optimizer.monitor_resource_usage()
    logger.info(f"  CPU: {resource_usage['cpu_percent']:.1f}%")
    logger.info(f"  Memory: {resource_usage['memory_percent']:.1f}% ({resource_usage['memory_used_gb']:.1f}GB)")
    
    if resource_usage['gpu_usage']:
        for gpu in resource_usage['gpu_usage']:
            logger.info(f"  GPU {gpu['device']}: {gpu['memory_percent']:.1f}% ({gpu['memory_used_gb']:.1f}GB)")
    
    # Get optimization recommendations
    recommendations = optimizer.get_optimization_recommendations()
    logger.info("\n💡 Optimization Recommendations:")
    for rec in recommendations:
        logger.info(f"  {rec['type'].upper()}: {rec['message']}")
    
    return optimized_config

async def federated_learning_example():
    """Demonstrate federated learning with multiple clients."""
    logger = logging.getLogger(__name__)
    logger.info("🌐 Starting federated learning example")
    
    # Create privacy configuration
    privacy_config = PrivacyConfig(
        epsilon=2.0,
        delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=0.8
    )
    
    # Create federated configuration
    federated_config = FederatedConfig(
        num_clients=5,
        min_clients=3,
        client_fraction=0.8,
        local_epochs=2,
        aggregation_method=AggregationMethod.FEDERATED_AVERAGING,
        client_privacy_budget=0.5,
        server_privacy_budget=1.0,
        communication_rounds=5
    )
    
    logger.info(f"Federated config: {federated_config.num_clients} clients, "
               f"{federated_config.communication_rounds} rounds")
    
    # Initialize federated trainer
    fed_trainer = FederatedPrivateTrainer(
        privacy_config=privacy_config,
        federated_config=federated_config
    )
    
    # Create mock client data paths (in practice, these would be real data)
    client_data_paths = {
        f"client_{i}": f"data/client_{i}/dataset.jsonl"
        for i in range(federated_config.num_clients)
    }
    
    logger.info("📡 Starting federated training simulation...")
    
    # Run federated training
    timer_id = performance_monitor.start_timer("federated_training_demo")
    
    try:
        results = await fed_trainer.train_federated(
            rounds=federated_config.communication_rounds,
            client_data_paths=client_data_paths,
            evaluation_data="data/test_dataset.jsonl"
        )
        
        performance_monitor.end_timer(timer_id)
        
        # Display results
        logger.info("🎉 Federated training completed!")
        logger.info("📊 Training Results:")
        logger.info(f"  Rounds completed: {results['rounds_completed']}")
        logger.info(f"  Final accuracy: {results['final_accuracy']:.4f}")
        logger.info(f"  Privacy spent: ε={results['privacy_spent']['epsilon']:.6f}")
        logger.info(f"  Average client privacy: ε={results['client_privacy_summary']['average_epsilon']:.6f}")
        logger.info(f"  Communication efficiency: {results['communication_efficiency']['total_updates']} updates")
        
        return results
        
    except Exception as e:
        performance_monitor.end_timer(timer_id)
        logger.error(f"Federated training failed: {e}")
        raise

def performance_monitoring_example():
    """Demonstrate advanced performance monitoring."""
    logger = logging.getLogger(__name__)
    logger.info("📊 Starting performance monitoring example")
    
    # Initialize resource optimizer for monitoring
    optimizer = ResourceOptimizer()
    
    # Simulate some workload monitoring
    logger.info("Monitoring resource usage during simulated workload...")
    
    import time
    for i in range(5):
        # Simulate some work
        time.sleep(1)
        
        # Monitor resources
        usage = optimizer.monitor_resource_usage()
        
        logger.info(f"Measurement {i+1}: CPU {usage['cpu_percent']:.1f}%, "
                   f"Memory {usage['memory_percent']:.1f}%")
    
    # Get recommendations based on monitoring
    recommendations = optimizer.get_optimization_recommendations()
    
    logger.info("\n🎯 Performance Analysis Results:")
    for rec in recommendations:
        level = rec['type'].upper()
        message = rec['message']
        logger.info(f"  [{level}] {message}")
    
    # Display optimization history
    if optimizer.optimization_history:
        logger.info(f"\n📈 Optimization History: {len(optimizer.optimization_history)} configurations stored")

def scaling_best_practices_demo():
    """Demonstrate scaling best practices and patterns."""
    logger = logging.getLogger(__name__)
    logger.info("🎓 Scaling Best Practices Demonstration")
    
    # Resource-aware configuration
    resource_profile = ResourceProfile.detect_system_profile()
    resource_score = resource_profile.get_resource_score()
    
    logger.info(f"System resource score: {resource_score:.1f}")
    
    # Scaling recommendations based on resources
    if resource_score < 50:
        scaling_strategy = "single_node_optimized"
        recommendations = [
            "Use gradient checkpointing to reduce memory usage",
            "Enable mixed precision training",
            "Optimize batch size for available memory",
            "Use efficient data loading with fewer workers"
        ]
    elif resource_score < 200:
        scaling_strategy = "multi_gpu_local"
        recommendations = [
            "Enable data parallelism across available GPUs",
            "Use gradient accumulation for larger effective batch sizes",
            "Implement model sharding for very large models",
            "Optimize inter-GPU communication"
        ]
    else:
        scaling_strategy = "distributed_multi_node"
        recommendations = [
            "Implement distributed training across nodes",
            "Use federated learning for privacy-preserving distributed training",
            "Optimize network communication with gradient compression",
            "Implement fault tolerance and recovery mechanisms"
        ]
    
    logger.info(f"Recommended scaling strategy: {scaling_strategy}")
    logger.info("Best practices for your system:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  {i}. {rec}")
    
    # Privacy-scaling trade-offs
    logger.info("\n🔒 Privacy-Scaling Trade-offs:")
    privacy_levels = [
        ("High Privacy (ε < 1.0)", "More noise, slower convergence, requires larger batch sizes"),
        ("Medium Privacy (ε = 1.0-5.0)", "Balanced trade-off, good for most applications"),
        ("Lower Privacy (ε > 5.0)", "Less noise, faster convergence, smaller batch sizes OK")
    ]
    
    for level, description in privacy_levels:
        logger.info(f"  • {level}: {description}")

async def main():
    """Run all scaling and optimization examples."""
    
    # Setup advanced logging
    setup_privacy_logging(
        log_level="INFO",
        log_file="logs/scaling_demo.log",
        structured_logging=True,
        privacy_redaction=True
    )
    
    logger = logging.getLogger(__name__)
    
    print("Privacy-Preserving Agent Finetuner - Advanced Scaling Examples")
    print("=" * 70)
    
    try:
        # 1. Resource Optimization
        print("\n1. 🚀 Resource Optimization and Auto-Configuration")
        print("-" * 50)
        optimized_config = resource_optimization_example()
        
        # 2. Performance Monitoring
        print("\n2. 📊 Advanced Performance Monitoring")
        print("-" * 50)
        performance_monitoring_example()
        
        # 3. Federated Learning
        print("\n3. 🌐 Federated Learning Simulation") 
        print("-" * 50)
        federated_results = await federated_learning_example()
        
        # 4. Scaling Best Practices
        print("\n4. 🎓 Scaling Best Practices")
        print("-" * 50)
        scaling_best_practices_demo()
        
        print("\n✅ All scaling examples completed successfully!")
        print("\nNext steps for production:")
        print("- Install full ML dependencies: pip install torch transformers datasets opacus")
        print("- Configure distributed training infrastructure")
        print("- Set up monitoring and alerting systems")
        print("- Implement automated scaling policies")
        print("- Test federated learning with real clients")
        
        return 0
        
    except Exception as e:
        logger.error(f"Scaling examples failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))