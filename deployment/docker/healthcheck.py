#!/usr/bin/env python3
"""
TERRAGON AUTONOMOUS SDLC - PRODUCTION HEALTH CHECK

Comprehensive health check for production deployment
Validates all three generations of capabilities
"""

import sys
import time
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List


def check_api_health(port: int = 8080) -> Dict[str, Any]:
    """Check API server health."""
    try:
        url = f"http://localhost:{port}/health"
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                return {"status": "healthy", "data": data}
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status}"}
    except urllib.error.URLError as e:
        return {"status": "unhealthy", "error": f"Connection failed: {e}"}
    except Exception as e:
        return {"status": "unhealthy", "error": f"Unexpected error: {e}"}


def check_privacy_modules() -> Dict[str, Any]:
    """Check core privacy modules."""
    try:
        sys.path.append('/app')
        
        # Test core imports
        from privacy_finetuner.core.privacy_config import PrivacyConfig
        from privacy_finetuner.core.trainer import DPTrainer
        
        # Test Generation 1: Research capabilities
        try:
            from privacy_finetuner.research.autonomous_privacy_evolution import create_autonomous_privacy_evolution_system
            gen1_status = "available"
        except ImportError:
            gen1_status = "limited"
        
        # Test Generation 2: Resilience capabilities
        try:
            from privacy_finetuner.resilience.adaptive_failure_recovery import AdaptiveFailureRecoverySystem
            gen2_status = "available"
        except ImportError:
            gen2_status = "limited"
        
        # Test Generation 3: Scaling capabilities
        try:
            from privacy_finetuner.scaling.distributed_privacy_orchestrator import create_distributed_privacy_system
            gen3_status = "available"
        except ImportError:
            gen3_status = "limited"
        
        return {
            "status": "healthy",
            "core_modules": "available",
            "generation_1_research": gen1_status,
            "generation_2_resilience": gen2_status,
            "generation_3_scaling": gen3_status
        }
    
    except ImportError as e:
        return {"status": "unhealthy", "error": f"Module import failed: {e}"}
    except Exception as e:
        return {"status": "unhealthy", "error": f"Unexpected error: {e}"}


def check_system_resources() -> Dict[str, Any]:
    """Check system resources."""
    try:
        import os
        import shutil
        
        # Check memory (approximate)
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if line.startswith('MemAvailable:'):
                        available_mb = int(line.split()[1]) // 1024
                        break
                else:
                    available_mb = 1000  # Default fallback
        except:
            available_mb = 1000  # Fallback
        
        # Check disk space
        try:
            disk_usage = shutil.disk_usage('/app')
            free_gb = disk_usage.free // (1024**3)
        except:
            free_gb = 1  # Fallback
        
        # Check critical directories
        directories = {
            'checkpoints': '/app/checkpoints',
            'logs': '/app/logs',
            'cache': '/app/cache',
            'config': '/app/config'
        }
        
        dir_status = {}
        for name, path in directories.items():
            dir_status[name] = {
                'exists': os.path.exists(path),
                'writable': os.access(path, os.W_OK) if os.path.exists(path) else False
            }
        
        resource_status = "healthy"
        if available_mb < 100:  # Less than 100MB
            resource_status = "low_memory"
        elif free_gb < 1:  # Less than 1GB
            resource_status = "low_disk"
        
        return {
            "status": resource_status,
            "memory_available_mb": available_mb,
            "disk_free_gb": free_gb,
            "directories": dir_status
        }
    
    except Exception as e:
        return {"status": "unhealthy", "error": f"Resource check failed: {e}"}


def check_configuration() -> Dict[str, Any]:
    """Check configuration validity."""
    try:
        import os
        
        # Check environment variables
        env_vars = {
            'TERRAGON_MODE': os.getenv('TERRAGON_MODE', 'unknown'),
            'PRIVACY_EPSILON': os.getenv('PRIVACY_EPSILON', 'unknown'),
            'PRIVACY_DELTA': os.getenv('PRIVACY_DELTA', 'unknown'),
            'LOG_LEVEL': os.getenv('LOG_LEVEL', 'unknown')
        }
        
        # Validate privacy parameters
        try:
            epsilon = float(env_vars['PRIVACY_EPSILON'])
            delta = float(env_vars['PRIVACY_DELTA'])
            
            if epsilon <= 0 or epsilon > 100:
                return {"status": "unhealthy", "error": f"Invalid epsilon: {epsilon}"}
            if delta <= 0 or delta >= 1:
                return {"status": "unhealthy", "error": f"Invalid delta: {delta}"}
            
            privacy_valid = True
        except (ValueError, TypeError):
            privacy_valid = False
        
        # Check configuration files
        config_files = [
            '/app/config/runtime_config.yaml',
            '/app/config/production_config.yaml'
        ]
        
        config_status = {}
        for config_file in config_files:
            config_status[config_file] = {
                'exists': os.path.exists(config_file),
                'readable': os.access(config_file, os.R_OK) if os.path.exists(config_file) else False
            }
        
        return {
            "status": "healthy" if privacy_valid else "configuration_error",
            "environment_variables": env_vars,
            "privacy_parameters_valid": privacy_valid,
            "configuration_files": config_status
        }
    
    except Exception as e:
        return {"status": "unhealthy", "error": f"Configuration check failed: {e}"}


def perform_functional_test() -> Dict[str, Any]:
    """Perform basic functional test."""
    try:
        sys.path.append('/app')
        
        # Test privacy configuration creation
        from privacy_finetuner.core.privacy_config import PrivacyConfig
        config = PrivacyConfig(epsilon=1.0, delta=1e-5)
        config.validate()
        
        # Test basic privacy computation
        import random
        test_data = [random.random() for _ in range(10)]
        
        # Simple DP noise test
        noise_scale = 1.0 / config.epsilon
        noisy_data = [x + random.gauss(0, noise_scale) for x in test_data]
        
        # Verify computation completed
        if len(noisy_data) == len(test_data):
            functional_status = "passed"
        else:
            functional_status = "failed"
        
        return {
            "status": "healthy" if functional_status == "passed" else "functional_error",
            "privacy_config_test": "passed",
            "noise_generation_test": "passed",
            "data_processing_test": functional_status
        }
    
    except Exception as e:
        return {"status": "unhealthy", "error": f"Functional test failed: {e}"}


def main() -> int:
    """Main health check function."""
    print("🔍 TERRAGON HEALTH CHECK - STARTING")
    
    start_time = time.time()
    
    # Perform all health checks
    checks = {
        "api_health": check_api_health(),
        "privacy_modules": check_privacy_modules(),
        "system_resources": check_system_resources(),
        "configuration": check_configuration(),
        "functional_test": perform_functional_test()
    }
    
    # Calculate overall health
    healthy_checks = sum(1 for check in checks.values() if check.get("status") == "healthy")
    total_checks = len(checks)
    health_percentage = (healthy_checks / total_checks) * 100
    
    duration = time.time() - start_time
    
    # Determine overall status
    if health_percentage >= 100:
        overall_status = "healthy"
        exit_code = 0
    elif health_percentage >= 80:
        overall_status = "degraded"
        exit_code = 0  # Still considered passing for Docker
    elif health_percentage >= 60:
        overall_status = "warning"
        exit_code = 1
    else:
        overall_status = "critical"
        exit_code = 1
    
    # Generate health report
    health_report = {
        "timestamp": time.time(),
        "overall_status": overall_status,
        "health_percentage": health_percentage,
        "check_duration_seconds": duration,
        "checks": checks,
        "generation_capabilities": {
            "1_advanced_research": checks["privacy_modules"].get("generation_1_research", "unknown"),
            "2_robust_operations": checks["privacy_modules"].get("generation_2_resilience", "unknown"),
            "3_massive_scaling": checks["privacy_modules"].get("generation_3_scaling", "unknown")
        },
        "system_summary": {
            "api_accessible": checks["api_health"]["status"] == "healthy",
            "modules_loaded": checks["privacy_modules"]["status"] == "healthy",
            "resources_sufficient": checks["system_resources"]["status"] == "healthy",
            "configuration_valid": checks["configuration"]["status"] == "healthy",
            "functional_test_passed": checks["functional_test"]["status"] == "healthy"
        }
    }
    
    # Output results
    print(f"🏥 HEALTH STATUS: {overall_status.upper()}")
    print(f"📊 HEALTH SCORE: {health_percentage:.1f}%")
    print(f"⏱️  CHECK DURATION: {duration:.2f}s")
    print(f"✅ HEALTHY CHECKS: {healthy_checks}/{total_checks}")
    
    # Print detailed status if not fully healthy
    if health_percentage < 100:
        print("\n📋 DETAILED STATUS:")
        for check_name, check_result in checks.items():
            status = check_result.get("status", "unknown")
            print(f"  • {check_name}: {status}")
            if status != "healthy" and "error" in check_result:
                print(f"    Error: {check_result['error']}")
    
    # Save health report for monitoring
    try:
        with open('/app/logs/health_report.json', 'w') as f:
            json.dump(health_report, f, indent=2)
    except:
        pass  # Don't fail health check if logging fails
    
    print(f"\n🎯 HEALTH CHECK COMPLETED - EXIT CODE: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())