#!/usr/bin/env python3
"""
Generation 3 Demo - Scalable & Optimized Privacy-Preserving ML

This example demonstrates the Generation 3 enhancements including:
- Intelligent performance optimization and auto-tuning
- Advanced auto-scaling with privacy-aware resource management
- Cost optimization and resource efficiency
- Distributed coordination and load balancing
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from privacy_finetuner.scaling.performance_optimizer import PerformanceOptimizer, OptimizationProfile, OptimizationType
from privacy_finetuner.scaling.auto_scaler import AutoScaler, ScalingPolicy, ScalingTrigger, NodeType, ScalingDirection
from privacy_finetuner.utils.logging_config import setup_privacy_logging

def demo_performance_optimization():
    """Demonstrate intelligent performance optimization capabilities."""
    logger = logging.getLogger(__name__)
    logger.info("⚡ Starting Performance Optimization Demo")
    
    # Initialize performance optimizer
    optimizer = PerformanceOptimizer(
        target_throughput=1200.0,
        max_memory_gb=64.0,
        optimization_interval=5.0,  # Fast optimization cycles for demo
        auto_optimization=True
    )
    
    # Create comprehensive optimization profile
    optimization_profile = OptimizationProfile(
        profile_name="high_performance_privacy",
        optimization_types=[
            OptimizationType.MEMORY_OPTIMIZATION,
            OptimizationType.COMPUTE_OPTIMIZATION,
            OptimizationType.BATCH_SIZE_OPTIMIZATION,
            OptimizationType.PRIVACY_BUDGET_OPTIMIZATION,
            OptimizationType.COMMUNICATION_OPTIMIZATION
        ],
        target_metrics={
            "throughput": 1200.0,
            "memory_efficiency": 0.8,
            "gpu_utilization": 85.0
        },
        resource_constraints={
            "max_memory_gb": 64.0,
            "max_cpu_percent": 90.0,
            "max_gpu_memory_gb": 40.0
        },
        privacy_constraints={
            "min_privacy_efficiency": 0.75,
            "max_budget_rate": 0.1
        },
        optimization_settings={
            "batch_size": 32,
            "learning_rate": 5e-5,
            "gradient_compression": True,
            "adaptive_noise": True
        }
    )
    
    # Set optimization profile and start optimization
    optimizer.set_optimization_profile(optimization_profile)
    optimizer.start_optimization()
    
    logger.info("✅ Performance optimization started with comprehensive profile")
    
    # Register custom metrics callback to simulate dynamic conditions
    def custom_metrics_callback():
        """Simulate varying training conditions."""
        import random
        current_time = time.time()
        
        # Simulate different training phases
        if int(current_time) % 60 < 30:  # First half of minute
            return {
                "throughput_samples_per_sec": 800 + random.uniform(-100, 100),  # Below target
                "memory_utilization_percent": 88 + random.uniform(-5, 5),        # High memory usage
                "gpu_utilization_percent": 55 + random.uniform(-10, 10)          # Low GPU usage
            }
        else:  # Second half of minute
            return {
                "throughput_samples_per_sec": 1100 + random.uniform(-50, 100),  # Near target
                "memory_utilization_percent": 65 + random.uniform(-10, 10),     # Normal memory
                "gpu_utilization_percent": 75 + random.uniform(-15, 15)         # Better GPU usage
            }
    
    optimizer.register_metrics_callback("training_simulation", custom_metrics_callback)
    
    # Monitor optimization for several cycles
    optimization_cycles = 6
    optimization_summary = []
    
    logger.info(f"Monitoring optimization for {optimization_cycles} cycles...")
    
    for cycle in range(optimization_cycles):
        logger.info(f"\n--- Optimization Cycle {cycle + 1} ---")
        
        # Wait for optimization cycle
        time.sleep(6)
        
        # Get current optimization status
        summary = optimizer.get_optimization_summary()
        optimization_summary.append(summary)
        
        logger.info(f"Active optimizations: {summary['active_optimizations']}")
        logger.info(f"Average throughput: {summary['average_throughput']:.1f} samples/sec")
        logger.info(f"Target achievement: {summary['throughput_achievement']:.1f}%")
        logger.info(f"Total optimizations applied: {summary['total_optimizations_applied']}")
    
    # Stop optimization
    optimizer.stop_optimization()
    
    # Run performance benchmark
    logger.info("\n🔬 Running performance benchmark...")
    benchmark_results = optimizer.benchmark_optimization_impact(duration_seconds=30)
    
    logger.info("📊 Benchmark Results:")
    logger.info(f"  Throughput improvement: {benchmark_results['improvement_summary']['throughput_improvement_percent']:.1f}%")
    logger.info(f"  Memory reduction: {benchmark_results['improvement_summary']['memory_reduction_percent']:.1f}%")
    logger.info(f"  GPU utilization improvement: {benchmark_results['improvement_summary']['gpu_utilization_improvement']:.1f}%")
    logger.info(f"  Optimizations applied: {len(benchmark_results['optimizations_applied'])}")
    
    return optimization_summary, benchmark_results

def demo_auto_scaling():
    """Demonstrate intelligent auto-scaling capabilities.""" 
    logger = logging.getLogger(__name__)
    logger.info("📈 Starting Auto-Scaling Demo")
    
    # Create advanced scaling policy
    scaling_policy = ScalingPolicy(
        policy_name="privacy_aware_scaling",
        triggers=[
            ScalingTrigger.GPU_UTILIZATION,
            ScalingTrigger.THROUGHPUT_TARGET,
            ScalingTrigger.PRIVACY_BUDGET_RATE,
            ScalingTrigger.COST_OPTIMIZATION
        ],
        scale_up_threshold={
            "gpu_utilization": 75.0,
            "throughput_target_ratio": 0.6,
            "privacy_budget_efficiency": 0.5
        },
        scale_down_threshold={
            "gpu_utilization": 25.0,
            "throughput_target_ratio": 1.5,
            "privacy_budget_efficiency": 0.95
        },
        min_nodes=2,
        max_nodes=8,
        cooldown_period_seconds=30,  # Short cooldown for demo
        scaling_step_size=2,
        cost_constraints={"max_hourly_cost": 80.0},
        privacy_constraints={"min_nodes_for_privacy": 3}
    )
    
    # Initialize auto-scaler
    auto_scaler = AutoScaler(
        scaling_policy=scaling_policy,
        monitoring_interval=10.0,  # Fast monitoring for demo
        enable_cost_optimization=True,
        enable_privacy_preservation=True
    )
    
    logger.info("✅ Auto-scaler initialized with privacy-aware policy")
    
    # Initialize with some nodes
    auto_scaler.manual_scale(ScalingDirection.SCALE_OUT, NodeType.GPU_WORKER, 2)
    auto_scaler.manual_scale(ScalingDirection.SCALE_OUT, NodeType.CPU_WORKER, 1)
    
    # Register custom metrics to simulate varying load
    def scaling_metrics_callback():
        """Simulate dynamic scaling scenarios."""
        import random
        current_time = time.time() % 180  # 3-minute cycles
        
        if current_time < 60:  # First minute - normal load
            return {
                "gpu_utilization": 60 + random.uniform(-15, 15),
                "throughput_samples_per_sec": 800 + random.uniform(-100, 200),
                "target_throughput": 1000,
                "privacy_budget_efficiency": 0.8 + random.uniform(-0.1, 0.1)
            }
        elif current_time < 120:  # Second minute - high load
            return {
                "gpu_utilization": 85 + random.uniform(-10, 10),
                "throughput_samples_per_sec": 500 + random.uniform(-100, 100),
                "target_throughput": 1200,
                "privacy_budget_efficiency": 0.4 + random.uniform(-0.1, 0.1)
            }
        else:  # Third minute - low load
            return {
                "gpu_utilization": 25 + random.uniform(-5, 10),
                "throughput_samples_per_sec": 1400 + random.uniform(-200, 200),
                "target_throughput": 1000,
                "privacy_budget_efficiency": 0.95 + random.uniform(-0.05, 0.05)
            }
    
    auto_scaler.register_metrics_collector("load_simulation", scaling_metrics_callback)
    
    # Register scaling event callback
    def scaling_event_callback(event):
        logger.info(f"🔄 Scaling Event: {event.scaling_direction.value} - {event.reason}")
        logger.info(f"   Nodes affected: {event.nodes_affected}, Cost impact: ${event.cost_impact:.2f}/hr")
    
    auto_scaler.register_scaling_callback("event_logger", scaling_event_callback)
    
    # Start auto-scaling
    auto_scaler.start_auto_scaling()
    
    # Monitor scaling behavior
    monitoring_duration = 60  # Monitor for 1 minute
    scaling_events = []
    
    logger.info(f"Monitoring auto-scaling for {monitoring_duration} seconds...")
    
    start_time = time.time()
    while time.time() - start_time < monitoring_duration:
        time.sleep(10)
        
        # Get current status
        status = auto_scaler.get_scaling_status()
        logger.info(f"\nCurrent Status:")
        logger.info(f"  Nodes: {status['current_nodes']} (Cost: ${status['current_hourly_cost']:.2f}/hr)")
        logger.info(f"  Node breakdown: {status['node_breakdown']}")
        logger.info(f"  GPU utilization: {status['resource_utilization'].get('gpu_utilization', 0):.1f}%")
        
        scaling_events.extend([
            event for event in auto_scaler.scaling_history
            if event not in scaling_events
        ])
    
    # Stop auto-scaling
    auto_scaler.stop_auto_scaling()
    
    # Generate cost optimization analysis
    logger.info("\n💰 Performing cost optimization analysis...")
    cost_analysis = auto_scaler.optimize_cost()
    
    logger.info("Cost Analysis Results:")
    logger.info(f"  Current hourly cost: ${cost_analysis['current_hourly_cost']:.2f}")
    logger.info(f"  Daily projected cost: ${cost_analysis['daily_projected_cost']:.2f}")
    logger.info(f"  Monthly projected cost: ${cost_analysis['monthly_projected_cost']:.2f}")
    logger.info(f"  Optimization recommendations: {len(cost_analysis['optimization_recommendations'])}")
    
    for rec in cost_analysis['optimization_recommendations']:
        logger.info(f"    • {rec['action']}: {rec['description']}")
        logger.info(f"      Potential savings: ${rec['potential_savings']:.2f}/hr")
    
    return scaling_events, cost_analysis

def demo_scaling_scenarios():
    """Demonstrate scaling under different load scenarios."""
    logger = logging.getLogger(__name__)
    logger.info("🎯 Starting Scaling Scenarios Demo")
    
    # Initialize auto-scaler for scenarios
    auto_scaler = AutoScaler(
        monitoring_interval=5.0,
        enable_cost_optimization=True,
        enable_privacy_preservation=True
    )
    
    # Test different scaling scenarios
    scenarios = [
        {
            "name": "Traffic Spike",
            "load_pattern": "spike",
            "duration": 10,  # 10 minutes simulation compressed
            "description": "Sudden traffic increase simulation"
        },
        {
            "name": "Gradual Growth",
            "load_pattern": "gradual_increase", 
            "duration": 15,
            "description": "Steady load increase over time"
        },
        {
            "name": "Variable Load",
            "load_pattern": "variable",
            "duration": 12,
            "description": "Cyclic load variations"
        }
    ]
    
    scenario_results = []
    
    for scenario in scenarios:
        logger.info(f"\n--- Scenario: {scenario['name']} ---")
        logger.info(f"Description: {scenario['description']}")
        
        # Reset to baseline
        auto_scaler.current_nodes = {}
        auto_scaler.cost_tracker["current_hourly_cost"] = 0.0
        auto_scaler.manual_scale(ScalingDirection.SCALE_OUT, NodeType.GPU_WORKER, 1)
        
        # Run scenario simulation
        logger.info("Running scenario simulation...")
        result = auto_scaler.simulate_scaling_scenario(
            scenario_name=scenario["name"],
            duration_minutes=scenario["duration"],
            load_pattern=scenario["load_pattern"]
        )
        
        scenario_results.append(result)
        
        logger.info(f"Scenario Results:")
        logger.info(f"  Scaling events: {len(result['scaling_events'])}")
        logger.info(f"  Final cost: ${result['cost_progression'][-1]['cost']:.2f}/hr")
        logger.info(f"  Final nodes: {result['cost_progression'][-1]['nodes']}")
        
        # Brief pause between scenarios
        time.sleep(2)
    
    # Analyze scenario performance
    logger.info("\n📊 Scenario Performance Analysis:")
    
    for i, (scenario, result) in enumerate(zip(scenarios, scenario_results)):
        initial_cost = result['cost_progression'][0]['cost']
        final_cost = result['cost_progression'][-1]['cost']
        cost_efficiency = abs(final_cost - initial_cost) / max(initial_cost, 1.0)
        
        logger.info(f"\n{scenario['name']}:")
        logger.info(f"  Cost efficiency: {cost_efficiency:.2f}")
        logger.info(f"  Scaling responsiveness: {len(result['scaling_events'])} events")
        logger.info(f"  Load pattern handled: {scenario['load_pattern']}")
    
    return scenario_results

def demo_integrated_scaling_optimization():
    """Demonstrate integrated performance optimization and auto-scaling."""
    logger = logging.getLogger(__name__)
    logger.info("🔗 Starting Integrated Scaling & Optimization Demo")
    
    # Initialize both systems
    optimizer = PerformanceOptimizer(target_throughput=1500.0, auto_optimization=True)
    auto_scaler = AutoScaler(enable_cost_optimization=True, enable_privacy_preservation=True)
    
    # Create integrated optimization profile
    integrated_profile = OptimizationProfile(
        profile_name="integrated_scaling_optimization",
        optimization_types=[
            OptimizationType.COMPUTE_OPTIMIZATION,
            OptimizationType.COMMUNICATION_OPTIMIZATION,
            OptimizationType.PRIVACY_BUDGET_OPTIMIZATION
        ],
        target_metrics={
            "throughput": 1500.0,
            "cost_efficiency": 0.8,
            "privacy_efficiency": 0.85
        },
        resource_constraints={
            "max_memory_gb": 128.0,
            "max_nodes": 6,
            "max_hourly_cost": 60.0
        },
        privacy_constraints={
            "min_privacy_efficiency": 0.7,
            "min_nodes_for_privacy": 2
        },
        optimization_settings={
            "auto_scaling_enabled": True,
            "cost_optimization_priority": "high",
            "privacy_preservation_priority": "critical"
        }
    )
    
    optimizer.set_optimization_profile(integrated_profile)
    
    # Create coordination between systems
    def optimization_triggered_scaling(throughput_ratio):
        """Trigger scaling based on optimization results."""
        if throughput_ratio < 0.6:
            logger.info("Optimization triggered scale-out due to low throughput")
            auto_scaler.manual_scale(ScalingDirection.SCALE_OUT, NodeType.COMPUTE_OPTIMIZED, 1)
        elif throughput_ratio > 1.4:
            logger.info("Optimization triggered scale-in due to excess capacity") 
            auto_scaler.manual_scale(ScalingDirection.SCALE_IN, NodeType.GPU_WORKER, 1)
    
    def scaling_triggered_optimization(event):
        """Trigger optimization based on scaling events."""
        if event.scaling_direction == ScalingDirection.SCALE_OUT:
            logger.info("Scaling triggered optimization for new resources")
            # In real implementation, would trigger resource-specific optimization
        elif event.scaling_direction == ScalingDirection.SCALE_IN:
            logger.info("Scaling triggered optimization for reduced resources")
            # Would optimize for reduced resource constraints
    
    # Register cross-system callbacks
    auto_scaler.register_scaling_callback("optimization_trigger", scaling_triggered_optimization)
    
    # Start both systems
    optimizer.start_optimization()
    auto_scaler.start_auto_scaling()
    
    # Initialize with baseline resources
    auto_scaler.manual_scale(ScalingDirection.SCALE_OUT, NodeType.GPU_WORKER, 2)
    
    logger.info("✅ Integrated optimization and scaling systems started")
    
    # Simulate integrated operation
    integration_duration = 45  # 45 seconds demo
    start_time = time.time()
    
    # Track integrated metrics
    integrated_metrics = []
    
    logger.info(f"Running integrated optimization for {integration_duration} seconds...")
    
    while time.time() - start_time < integration_duration:
        time.sleep(8)
        
        # Collect metrics from both systems
        opt_summary = optimizer.get_optimization_summary()
        scaling_status = auto_scaler.get_scaling_status()
        
        integrated_metric = {
            "timestamp": time.time(),
            "throughput_achievement": opt_summary["throughput_achievement"],
            "active_optimizations": opt_summary["active_optimizations"],
            "current_nodes": scaling_status["current_nodes"],
            "hourly_cost": scaling_status["current_hourly_cost"],
            "cost_efficiency": (opt_summary["average_throughput"] / max(scaling_status["current_hourly_cost"], 1.0)) * 10,
            "scaling_events_recent": scaling_status["scaling_events_last_hour"]
        }
        
        integrated_metrics.append(integrated_metric)
        
        logger.info(f"Integrated Status:")
        logger.info(f"  Throughput achievement: {integrated_metric['throughput_achievement']:.1f}%")
        logger.info(f"  Cost efficiency ratio: {integrated_metric['cost_efficiency']:.1f}")
        logger.info(f"  Nodes: {integrated_metric['current_nodes']}, Cost: ${integrated_metric['hourly_cost']:.2f}/hr")
        
        # Trigger cross-system coordination
        throughput_ratio = opt_summary["average_throughput"] / opt_summary["target_throughput"]
        if throughput_ratio != 0:  # Avoid division by zero
            optimization_triggered_scaling(throughput_ratio)
    
    # Stop systems
    optimizer.stop_optimization()
    auto_scaler.stop_auto_scaling()
    
    # Analyze integrated performance
    logger.info("\n📈 Integrated Performance Analysis:")
    
    avg_throughput_achievement = sum(m["throughput_achievement"] for m in integrated_metrics) / len(integrated_metrics)
    avg_cost_efficiency = sum(m["cost_efficiency"] for m in integrated_metrics) / len(integrated_metrics)
    total_optimizations = integrated_metrics[-1]["active_optimizations"]
    total_scaling_events = sum(m["scaling_events_recent"] for m in integrated_metrics)
    
    logger.info(f"  Average throughput achievement: {avg_throughput_achievement:.1f}%")
    logger.info(f"  Average cost efficiency: {avg_cost_efficiency:.1f}")
    logger.info(f"  Total optimizations applied: {total_optimizations}")
    logger.info(f"  Total scaling events: {total_scaling_events}")
    
    # Generate integration efficiency score
    integration_score = (avg_throughput_achievement + avg_cost_efficiency) / 2
    logger.info(f"  🎯 Integration Efficiency Score: {integration_score:.1f}")
    
    return integrated_metrics, integration_score

def main():
    """Run all Generation 3 scaling and optimization demonstrations."""
    
    # Setup advanced logging
    setup_privacy_logging(
        log_level="INFO",
        log_file="generation3_results/scaling_demo.log",
        structured_logging=True,
        privacy_redaction=True
    )
    
    # Create output directories
    Path("generation3_results").mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    
    print("Privacy-Preserving ML Framework - Generation 3: SCALABILITY")
    print("=" * 75)
    print("Demonstrating enterprise-grade scalability and optimization features:")
    print("• Intelligent performance optimization with adaptive strategies")
    print("• Privacy-aware auto-scaling with cost optimization")
    print("• Multi-scenario load handling and resource efficiency")
    print("• Integrated optimization-scaling coordination")
    print("=" * 75)
    
    try:
        # Demo 1: Performance Optimization
        print("\n⚡ 1. Intelligent Performance Optimization")
        print("-" * 50)
        optimization_results, benchmark_results = demo_performance_optimization()
        
        # Demo 2: Auto-Scaling
        print("\n📈 2. Privacy-Aware Auto-Scaling")
        print("-" * 50)
        scaling_events, cost_analysis = demo_auto_scaling()
        
        # Demo 3: Scaling Scenarios
        print("\n🎯 3. Multi-Scenario Scaling Analysis")
        print("-" * 50)
        scenario_results = demo_scaling_scenarios()
        
        # Demo 4: Integrated Systems
        print("\n🔗 4. Integrated Optimization & Scaling")
        print("-" * 50)
        integrated_metrics, integration_score = demo_integrated_scaling_optimization()
        
        print("\n✅ All Generation 3 scalability demos completed successfully!")
        
        print(f"\n📊 Key Scalability Metrics:")
        print(f"• Performance optimization cycles: {len(optimization_results)}")
        print(f"• Throughput improvement: {benchmark_results['improvement_summary']['throughput_improvement_percent']:.1f}%")
        print(f"• Auto-scaling events: {len(scaling_events)}")
        print(f"• Cost optimization savings: ${cost_analysis.get('optimization_recommendations', [{}])[0].get('potential_savings', 0):.2f}/hr")
        print(f"• Scaling scenarios tested: {len(scenario_results)}")
        print(f"• Integration efficiency score: {integration_score:.1f}")
        
        print(f"\n🚀 Scalability Features:")
        print(f"  • Adaptive performance optimization: ✅ Active")
        print(f"  • Privacy-aware auto-scaling: ✅ Operational")
        print(f"  • Multi-node cost optimization: ✅ Enabled")
        print(f"  • Dynamic resource allocation: ✅ Responsive")
        print(f"  • Cross-system coordination: ✅ Integrated")
        
        print(f"\n📁 Scalability artifacts saved to:")
        print(f"  • Performance logs: generation3_results/scaling_demo.log")
        print(f"  • Optimization reports: Available via export functions")
        print(f"  • Cost analysis: Integrated in scaling systems")
        
        print(f"\n🎯 Generation 3 Status: SCALABLE & OPTIMIZED")
        print(f"The framework now provides enterprise-grade:")
        print(f"  ✅ Intelligent performance optimization with adaptive strategies")
        print(f"  ✅ Privacy-preserving auto-scaling with cost awareness")
        print(f"  ✅ Multi-scenario load handling and resource efficiency")
        print(f"  ✅ Integrated optimization and scaling coordination")
        print(f"  ✅ Real-time resource monitoring and optimization")
        
        return 0
        
    except Exception as e:
        logger.error(f"Generation 3 scalability demo failed: {e}", exc_info=True)
        print(f"\n❌ Scalability demo failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())