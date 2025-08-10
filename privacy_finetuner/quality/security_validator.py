"""Security validation and compliance checking module."""

import os
import hashlib
import tempfile
import subprocess
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Comprehensive security validation for privacy-preserving ML."""
    
    def __init__(self):
        self.validation_rules = self._load_security_rules()
        self.severity_levels = ["low", "medium", "high", "critical"]
        
    def _load_security_rules(self) -> Dict[str, Any]:
        """Load security validation rules."""
        return {
            "code_injection": {
                "patterns": [r"exec\(", r"eval\(", r"__import__\("],
                "severity": "critical"
            },
            "hardcoded_secrets": {
                "patterns": [r"password\s*=\s*[\"'][^\"']+[\"']", r"api_key\s*=\s*[\"'][^\"']+[\"']"],
                "severity": "high"
            },
            "insecure_random": {
                "patterns": [r"random\.random\(", r"random\.randint\("],
                "severity": "medium"
            }
        }
    
    def validate_code_security(self, code_content: str) -> Dict[str, Any]:
        """Validate code for security vulnerabilities."""
        vulnerabilities = []
        
        for rule_name, rule_config in self.validation_rules.items():
            import re
            for pattern in rule_config["patterns"]:
                matches = re.findall(pattern, code_content, re.IGNORECASE)
                if matches:
                    vulnerabilities.append({
                        "rule": rule_name,
                        "severity": rule_config["severity"],
                        "matches": len(matches),
                        "pattern": pattern
                    })
        
        return {
            "total_vulnerabilities": len(vulnerabilities),
            "vulnerabilities": vulnerabilities,
            "passed": len(vulnerabilities) == 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def validate_privacy_implementation(self, privacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate privacy implementation meets standards."""
        issues = []
        
        # Check epsilon bounds
        epsilon = privacy_config.get("epsilon", 0)
        if epsilon <= 0:
            issues.append({"type": "invalid_epsilon", "severity": "critical"})
        elif epsilon > 10:
            issues.append({"type": "high_epsilon", "severity": "medium"})
        
        # Check delta bounds
        delta = privacy_config.get("delta", 0)
        if delta <= 0 or delta >= 1:
            issues.append({"type": "invalid_delta", "severity": "critical"})
        
        # Check noise multiplier
        noise_multiplier = privacy_config.get("noise_multiplier", 0)
        if noise_multiplier <= 0:
            issues.append({"type": "invalid_noise", "severity": "high"})
        
        return {
            "total_issues": len(issues),
            "issues": issues,
            "passed": len(issues) == 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def validate_dependencies(self) -> Dict[str, Any]:
        """Validate dependencies for known vulnerabilities."""
        vulnerabilities = []
        
        # Mock vulnerability check
        vulnerable_packages = ["insecure-package", "old-crypto"]
        
        try:
            import pkg_resources
            installed_packages = [pkg.project_name for pkg in pkg_resources.working_set]
            
            for pkg in installed_packages:
                if pkg in vulnerable_packages:
                    vulnerabilities.append({
                        "package": pkg,
                        "severity": "high",
                        "description": f"Known vulnerable package: {pkg}"
                    })
        except ImportError:
            logger.warning("pkg_resources not available for dependency checking")
        
        return {
            "total_vulnerabilities": len(vulnerabilities),
            "vulnerabilities": vulnerabilities,
            "passed": len(vulnerabilities) == 0,
            "timestamp": datetime.now().isoformat()
        }


class ComplianceChecker:
    """Compliance validation for privacy regulations."""
    
    def __init__(self):
        self.compliance_frameworks = {
            "GDPR": self._check_gdpr_compliance,
            "CCPA": self._check_ccpa_compliance,
            "HIPAA": self._check_hipaa_compliance,
            "PIPEDA": self._check_pipeda_compliance
        }
    
    def check_compliance(self, framework: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with specific framework."""
        if framework not in self.compliance_frameworks:
            return {
                "framework": framework,
                "supported": False,
                "error": f"Framework {framework} not supported"
            }
        
        return self.compliance_frameworks[framework](config)
    
    def check_all_frameworks(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with all supported frameworks."""
        results = {}
        
        for framework in self.compliance_frameworks:
            results[framework] = self.check_compliance(framework, config)
        
        total_passed = sum(1 for result in results.values() if result.get("compliant", False))
        
        return {
            "frameworks_checked": len(self.compliance_frameworks),
            "frameworks_passed": total_passed,
            "overall_compliance": total_passed == len(self.compliance_frameworks),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_gdpr_compliance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check GDPR compliance requirements."""
        requirements = []
        
        # Check data minimization
        if config.get("data_collection", {}).get("minimal", True):
            requirements.append({"name": "data_minimization", "passed": True})
        else:
            requirements.append({"name": "data_minimization", "passed": False})
        
        # Check consent management
        if config.get("consent_management", False):
            requirements.append({"name": "consent_management", "passed": True})
        else:
            requirements.append({"name": "consent_management", "passed": False})
        
        # Check right to erasure
        if config.get("right_to_erasure", False):
            requirements.append({"name": "right_to_erasure", "passed": True})
        else:
            requirements.append({"name": "right_to_erasure", "passed": False})
        
        passed_count = sum(1 for req in requirements if req["passed"])
        
        return {
            "framework": "GDPR",
            "requirements_checked": len(requirements),
            "requirements_passed": passed_count,
            "compliant": passed_count == len(requirements),
            "requirements": requirements,
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_ccpa_compliance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check CCPA compliance requirements."""
        requirements = []
        
        # Check privacy notice
        if config.get("privacy_notice", False):
            requirements.append({"name": "privacy_notice", "passed": True})
        else:
            requirements.append({"name": "privacy_notice", "passed": False})
        
        # Check opt-out mechanism
        if config.get("opt_out_mechanism", False):
            requirements.append({"name": "opt_out_mechanism", "passed": True})
        else:
            requirements.append({"name": "opt_out_mechanism", "passed": False})
        
        passed_count = sum(1 for req in requirements if req["passed"])
        
        return {
            "framework": "CCPA",
            "requirements_checked": len(requirements),
            "requirements_passed": passed_count,
            "compliant": passed_count == len(requirements),
            "requirements": requirements,
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_hipaa_compliance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check HIPAA compliance requirements."""
        requirements = []
        
        # Check encryption
        if config.get("encryption_at_rest", False) and config.get("encryption_in_transit", False):
            requirements.append({"name": "encryption", "passed": True})
        else:
            requirements.append({"name": "encryption", "passed": False})
        
        # Check access controls
        if config.get("access_controls", False):
            requirements.append({"name": "access_controls", "passed": True})
        else:
            requirements.append({"name": "access_controls", "passed": False})
        
        # Check audit logging
        if config.get("audit_logging", False):
            requirements.append({"name": "audit_logging", "passed": True})
        else:
            requirements.append({"name": "audit_logging", "passed": False})
        
        passed_count = sum(1 for req in requirements if req["passed"])
        
        return {
            "framework": "HIPAA",
            "requirements_checked": len(requirements),
            "requirements_passed": passed_count,
            "compliant": passed_count == len(requirements),
            "requirements": requirements,
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_pipeda_compliance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check PIPEDA compliance requirements."""
        requirements = []
        
        # Check purpose limitation
        if config.get("purpose_limitation", True):
            requirements.append({"name": "purpose_limitation", "passed": True})
        else:
            requirements.append({"name": "purpose_limitation", "passed": False})
        
        # Check data retention policy
        if config.get("data_retention_policy", False):
            requirements.append({"name": "data_retention_policy", "passed": True})
        else:
            requirements.append({"name": "data_retention_policy", "passed": False})
        
        passed_count = sum(1 for req in requirements if req["passed"])
        
        return {
            "framework": "PIPEDA",
            "requirements_checked": len(requirements),
            "requirements_passed": passed_count,
            "compliant": passed_count == len(requirements),
            "requirements": requirements,
            "timestamp": datetime.now().isoformat()
        }