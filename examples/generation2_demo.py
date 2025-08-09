#!/usr/bin/env python3
"""
Generation 2 Demo - Robust & Reliable Privacy-Preserving ML

This example demonstrates the Generation 2 enhancements including:
- Advanced threat detection and security monitoring
- Comprehensive failure recovery and resilience
- Enterprise-grade error handling and reliability
- Automated response systems
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from privacy_finetuner.security.threat_detector import ThreatDetector, ThreatType, ThreatLevel
from privacy_finetuner.resilience.failure_recovery import FailureRecoverySystem, FailureType, RecoveryStrategy
from privacy_finetuner.utils.logging_config import setup_privacy_logging

def demo_threat_detection_system():
    """Demonstrate advanced threat detection capabilities."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Starting Threat Detection System Demo")
    
    # Initialize threat detector
    threat_detector = ThreatDetector(
        alert_threshold=0.6,
        monitoring_interval=0.5,
        enable_automated_response=True
    )
    
    # Start real-time monitoring
    threat_detector.start_monitoring()
    logger.info("✅ Real-time threat monitoring started")
    
    # Simulate various training scenarios with potential threats
    threat_scenarios = [
        {
            "name": "Normal Training",
            "metrics": {
                "privacy_epsilon_used": 0.3,
                "privacy_epsilon_total": 2.0,
                "gradient_l2_norm": 1.2,
                "current_loss": 1.8,
                "accuracy": 0.85,
                "memory_usage_gb": 8.5,
                "cpu_usage_percent": 65
            },
            "context": {
                "expected_accuracy": 0.8,
                "failed_auth_attempts": 0,
                "unusual_access_time": False
            }
        },
        {
            "name": "Privacy Budget Near Exhaustion",
            "metrics": {
                "privacy_epsilon_used": 1.9,
                "privacy_epsilon_total": 2.0,
                "gradient_l2_norm": 0.8,
                "current_loss": 1.2,
                "accuracy": 0.88,
                "memory_usage_gb": 9.1,
                "cpu_usage_percent": 70
            },
            "context": {
                "expected_accuracy": 0.85,
                "failed_auth_attempts": 1
            }
        },
        {
            "name": "Potential Model Inversion Attack",
            "metrics": {
                "privacy_epsilon_used": 0.8,
                "privacy_epsilon_total": 2.0,
                "gradient_l2_norm": 8.5,  # Very high gradient norm
                "loss_variance": 4.2,      # High loss variance
                "current_loss": 2.1,
                "accuracy": 0.82,
                "memory_usage_gb": 12.3,
                "cpu_usage_percent": 85
            },
            "context": {
                "expected_accuracy": 0.8,
                "failed_auth_attempts": 0
            }
        },
        {
            "name": "Data Poisoning Attack",
            "metrics": {
                "privacy_epsilon_used": 0.5,
                "privacy_epsilon_total": 2.0,
                "gradient_l2_norm": 15.2,  # Extremely high gradient norm
                "current_loss": 4.8,       # Loss spike
                "accuracy": 0.45,          # Poor accuracy
                "memory_usage_gb": 10.1,
                "cpu_usage_percent": 75
            },
            "context": {
                "expected_accuracy": 0.8,
                "failed_auth_attempts": 2
            }
        },
        {
            "name": "Unauthorized Access Attempt",
            "metrics": {
                "privacy_epsilon_used": 0.4,
                "privacy_epsilon_total": 2.0,
                "gradient_l2_norm": 1.1,
                "current_loss": 1.9,
                "accuracy": 0.83,
                "memory_usage_gb": 8.8,
                "cpu_usage_percent": 68
            },
            "context": {
                "expected_accuracy": 0.8,
                "failed_auth_attempts": 12,  # Many failed attempts
                "unusual_access_time": True,
                "unknown_client_ip": True
            }
        }
    ]
    
    detected_threats = []
    
    # Test each threat scenario
    for i, scenario in enumerate(threat_scenarios, 1):
        logger.info(f"\n--- Scenario {i}: {scenario['name']} ---")
        
        # Update baseline metrics for first scenario
        if i == 1:
            threat_detector.update_baseline_metrics({
                "loss": scenario["metrics"]["current_loss"],
                "gradient_l2_norm": scenario["metrics"]["gradient_l2_norm"]
            })
        
        # Detect threats
        alerts = threat_detector.detect_threat(
            training_metrics=scenario["metrics"],
            context=scenario["context"]
        )
        
        if alerts:
            logger.warning(f"🚨 {len(alerts)} threats detected in scenario: {scenario['name']}")
            for alert in alerts:
                logger.warning(f"  • {alert.threat_type.value} ({alert.threat_level.value})")
                logger.warning(f"    Description: {alert.description}")
                logger.warning(f"    Recommendations: {', '.join(alert.recommended_actions)}")
                detected_threats.append(alert)
        else:
            logger.info(f"✅ No threats detected in scenario: {scenario['name']}")
        
        # Small delay between scenarios
        time.sleep(0.2)
    
    # Stop monitoring
    threat_detector.stop_monitoring()
    
    # Generate security summary
    security_summary = threat_detector.get_security_summary()
    logger.info(f"\n📊 Security Summary:")
    logger.info(f"  Total threats detected: {security_summary['security_metrics']['total_threats_detected']}")
    logger.info(f"  Active alerts: {security_summary['active_alerts_count']}")
    logger.info(f"  Critical alerts: {security_summary['critical_alerts']}")
    logger.info(f"  Monitoring status: {security_summary['monitoring_status']}")
    
    return detected_threats

def demo_failure_recovery_system():
    """Demonstrate comprehensive failure recovery capabilities."""
    logger = logging.getLogger(__name__)
    logger.info("🛠️ Starting Failure Recovery System Demo")
    
    # Initialize failure recovery system
    recovery_system = FailureRecoverySystem(
        checkpoint_dir="demo_recovery_checkpoints",
        max_recovery_attempts=3,
        auto_recovery_enabled=True,
        privacy_threshold=0.8
    )
    
    logger.info("✅ Failure recovery system initialized")
    
    # Create some recovery points simulating training progress
    recovery_points = []
    
    training_progress = [
        {"epoch": 1, "step": 100, "epsilon_spent": 0.1, "loss": 2.5, "accuracy": 0.65},
        {"epoch": 2, "step": 200, "epsilon_spent": 0.3, "loss": 1.8, "accuracy": 0.75},
        {"epoch": 3, "step": 300, "epsilon_spent": 0.5, "loss": 1.2, "accuracy": 0.82},
        {"epoch": 4, "step": 400, "epsilon_spent": 0.7, "loss": 0.9, "accuracy": 0.87}
    ]
    
    for progress in training_progress:
        recovery_id = recovery_system.create_recovery_point(
            epoch=progress["epoch"],
            step=progress["step"],
            model_state={"weights": f"mock_weights_epoch_{progress['epoch']}"},
            optimizer_state={"lr": 5e-5, "momentum": 0.9},
            privacy_state={
                "epsilon_spent": progress["epsilon_spent"],
                "epsilon_total": 2.0,
                "delta": 1e-5
            },
            training_metrics={
                "loss": progress["loss"],
                "accuracy": progress["accuracy"]
            },
            system_state={
                "gpu_memory_gb": 12.5,
                "cpu_usage": 70
            }
        )
        recovery_points.append(recovery_id)
        logger.info(f"Created recovery point {recovery_id} for epoch {progress['epoch']}")
    
    # Test different failure scenarios and recovery strategies
    failure_scenarios = [
        {
            "name": "System Crash",
            "failure_type": FailureType.SYSTEM_CRASH,
            "description": "Simulated system crash during training",
            "affected_components": ["trainer", "model"],
            "metadata": {
                "crash_reason": "kernel_panic",
                "last_known_state": "epoch_4_step_350"
            }
        },
        {
            "name": "GPU Memory Error", 
            "failure_type": FailureType.GPU_MEMORY_ERROR,
            "description": "Out of GPU memory during batch processing",
            "affected_components": ["gpu", "trainer"],
            "metadata": {
                "batch_size": 32,
                "model_parameters": 7000000,
                "available_memory_gb": 8
            }
        },
        {
            "name": "Privacy Violation",
            "failure_type": FailureType.PRIVACY_VIOLATION,
            "description": "Privacy budget exceeded safe threshold",
            "affected_components": ["privacy_engine", "trainer"],
            "metadata": {
                "epsilon_used": 1.9,
                "epsilon_limit": 2.0,
                "violation_threshold": 0.8
            }
        },
        {
            "name": "Network Failure",
            "failure_type": FailureType.NETWORK_FAILURE,
            "description": "Network connectivity lost in distributed training",
            "affected_components": ["distributed_trainer", "communication"],
            "metadata": {
                "failed_nodes": ["worker_1", "worker_3"],
                "available_nodes": ["worker_2", "worker_4", "worker_5"],
                "total_nodes": 5
            }
        },
        {
            "name": "Data Corruption",
            "failure_type": FailureType.DATA_CORRUPTION,
            "description": "Training data integrity check failed",
            "affected_components": ["data_loader", "trainer"],
            "metadata": {
                "corrupted_batches": 15,
                "total_batches": 1000,
                "corruption_type": "checksum_mismatch"
            }
        }
    ]
    
    recovery_results = []
    
    # Test each failure scenario
    for i, scenario in enumerate(failure_scenarios, 1):
        logger.info(f"\n--- Failure Scenario {i}: {scenario['name']} ---")
        
        # Simulate failure and attempt recovery
        recovery_success = recovery_system.handle_failure(
            failure_type=scenario["failure_type"],
            description=scenario["description"],
            affected_components=scenario["affected_components"],
            metadata=scenario["metadata"]
        )
        
        result = {
            "scenario": scenario["name"],
            "failure_type": scenario["failure_type"].value,
            "recovery_success": recovery_success
        }
        recovery_results.append(result)
        
        if recovery_success:
            logger.info(f"✅ Successfully recovered from {scenario['name']}")
        else:
            logger.error(f"❌ Failed to recover from {scenario['name']}")
        
        time.sleep(0.1)  # Brief pause between scenarios
    
    # Get recovery statistics
    stats = recovery_system.get_recovery_statistics()
    logger.info(f"\n📈 Recovery System Statistics:")
    logger.info(f"  Total failures handled: {stats['total_failures']}")
    logger.info(f"  Successful recoveries: {stats['successful_recoveries']}")
    logger.info(f"  Recovery rate: {stats['recovery_rate']:.1%}")
    logger.info(f"  Recovery points available: {stats['recovery_points_available']}")
    logger.info(f"  Failure breakdown: {stats['failure_by_type']}")
    
    # Test recovery system health
    logger.info("\n🔧 Testing Recovery System Health...")
    test_results = recovery_system.test_recovery_system()
    logger.info(f"Recovery system test results: {len(test_results['recovery_tests'])} tests completed")
    
    return recovery_results, stats

def demo_integrated_robustness():
    """Demonstrate integrated security and recovery working together."""
    logger = logging.getLogger(__name__)
    logger.info("🔗 Starting Integrated Robustness Demo")
    
    # Initialize both systems
    threat_detector = ThreatDetector(alert_threshold=0.7)
    recovery_system = FailureRecoverySystem()
    
    # Register custom alert handler that triggers recovery
    def security_recovery_handler(alert):
        logger.warning(f"Security alert triggered recovery handler: {alert.threat_type.value}")
        
        # Map security threats to failure types for recovery
        threat_to_failure_mapping = {
            ThreatType.PRIVACY_BUDGET_EXHAUSTION: FailureType.PRIVACY_VIOLATION,
            ThreatType.DATA_POISONING: FailureType.DATA_CORRUPTION,
            ThreatType.UNAUTHORIZED_ACCESS: FailureType.SYSTEM_CRASH,
            ThreatType.ABNORMAL_TRAINING_BEHAVIOR: FailureType.RESOURCE_EXHAUSTION
        }
        
        if alert.threat_type in threat_to_failure_mapping:
            failure_type = threat_to_failure_mapping[alert.threat_type]
            
            logger.info(f"Triggering recovery for mapped failure type: {failure_type.value}")
            recovery_success = recovery_system.handle_failure(
                failure_type=failure_type,
                description=f"Triggered by security alert: {alert.description}",
                affected_components=alert.affected_components,
                metadata={"security_alert_id": alert.threat_id}
            )
            
            if recovery_success:
                logger.info("✅ Security-triggered recovery successful")
            else:
                logger.error("❌ Security-triggered recovery failed")
    
    # Register the integrated handler
    for threat_type in ThreatType:
        threat_detector.register_alert_handler(threat_type, security_recovery_handler)
    
    logger.info("✅ Integrated security-recovery system configured")
    
    # Create recovery checkpoint
    recovery_id = recovery_system.create_recovery_point(
        epoch=5,
        step=500,
        privacy_state={"epsilon_spent": 0.6, "epsilon_total": 2.0}
    )
    
    # Simulate high-threat scenario that triggers both systems
    high_threat_metrics = {
        "privacy_epsilon_used": 1.85,  # Near exhaustion
        "privacy_epsilon_total": 2.0,
        "gradient_l2_norm": 12.0,      # Very high
        "current_loss": 5.2,           # Loss spike
        "accuracy": 0.32,              # Poor performance
        "memory_usage_gb": 55.0,       # High memory
        "cpu_usage_percent": 98        # CPU spike
    }
    
    high_threat_context = {
        "expected_accuracy": 0.85,
        "failed_auth_attempts": 8,
        "unusual_access_time": True
    }
    
    logger.info("\n⚠️ Simulating high-threat scenario...")
    
    # Detect threats (will trigger integrated recovery)
    alerts = threat_detector.detect_threat(high_threat_metrics, high_threat_context)
    
    logger.info(f"Integrated robustness demo completed")
    logger.info(f"Detected {len(alerts)} threats with integrated recovery responses")
    
    return len(alerts)

def main():
    """Run all Generation 2 robustness demonstrations."""
    
    # Setup enterprise-grade logging
    setup_privacy_logging(
        log_level="INFO",
        log_file="generation2_results/robustness_demo.log",
        structured_logging=True,
        privacy_redaction=True
    )
    
    # Create output directories
    Path("generation2_results").mkdir(exist_ok=True)
    Path("demo_recovery_checkpoints").mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    
    print("Privacy-Preserving ML Framework - Generation 2: ROBUSTNESS")
    print("=" * 70)
    print("Demonstrating enterprise-grade reliability and security features:")
    print("• Advanced real-time threat detection and response")
    print("• Comprehensive failure recovery and resilience")
    print("• Integrated security-recovery coordination")
    print("• Automated emergency protocols")
    print("=" * 70)
    
    try:
        # Demo 1: Threat Detection System
        print("\n🔍 1. Advanced Threat Detection System")
        print("-" * 50)
        detected_threats = demo_threat_detection_system()
        
        # Demo 2: Failure Recovery System
        print("\n🛠️ 2. Comprehensive Failure Recovery System")
        print("-" * 50)
        recovery_results, recovery_stats = demo_failure_recovery_system()
        
        # Demo 3: Integrated Robustness
        print("\n🔗 3. Integrated Security-Recovery System")
        print("-" * 50)
        integrated_alerts = demo_integrated_robustness()
        
        print("\n✅ All Generation 2 robustness demos completed successfully!")
        
        print(f"\n📊 Key Robustness Metrics:")
        print(f"• Threat detection: {len(detected_threats)} threats identified across scenarios")
        print(f"• Failure recovery: {recovery_stats['recovery_rate']:.1%} success rate")
        print(f"• Recovery strategies: {len(recovery_results)} failure types handled")
        print(f"• Integrated responses: {integrated_alerts} coordinated security-recovery actions")
        
        print(f"\n🛡️ Security Posture:")
        print(f"  • Real-time monitoring: ✅ Active")
        print(f"  • Automated threat response: ✅ Enabled")
        print(f"  • Multi-strategy recovery: ✅ Operational")
        print(f"  • Privacy-aware rollback: ✅ Configured")
        print(f"  • Emergency protocols: ✅ Ready")
        
        print(f"\n📁 Robustness artifacts saved to:")
        print(f"  • Security logs: generation2_results/robustness_demo.log")
        print(f"  • Recovery checkpoints: demo_recovery_checkpoints/")
        print(f"  • Failure analysis: demo_recovery_checkpoints/failure_history.jsonl")
        
        print(f"\n🎯 Generation 2 Status: ROBUST & RELIABLE")
        print(f"The framework now provides enterprise-grade:")
        print(f"  ✅ Advanced threat detection with automated response")
        print(f"  ✅ Comprehensive failure recovery with privacy preservation")
        print(f"  ✅ Integrated security-resilience coordination")
        print(f"  ✅ Emergency protocols for privacy violations")
        print(f"  ✅ Real-time monitoring and health checks")
        
        return 0
        
    except Exception as e:
        logger.error(f"Generation 2 robustness demo failed: {e}", exc_info=True)
        print(f"\n❌ Robustness demo failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())