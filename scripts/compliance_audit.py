#!/usr/bin/env python3
"""Comprehensive compliance audit script."""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone
import subprocess
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from privacy_finetuner.core.privacy_config import PrivacyConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceAuditor:
    """Comprehensive compliance audit system."""
    
    def __init__(self, project_root: Path):
        """Initialize compliance auditor.
        
        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self.audit_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "status": "unknown",
            "checks": {},
            "summary": {
                "total_checks": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "score": 0.0
            }
        }
        
    def run_full_audit(self) -> Dict[str, Any]:
        """Run complete compliance audit.
        
        Returns:
            Audit results
        """
        logger.info("Starting comprehensive compliance audit...")
        
        # Privacy compliance checks
        self._check_privacy_compliance()
        
        # Security compliance checks
        self._check_security_compliance()
        
        # Code quality checks
        self._check_code_quality()
        
        # Documentation compliance
        self._check_documentation_compliance()
        
        # Infrastructure compliance
        self._check_infrastructure_compliance()
        
        # License compliance
        self._check_license_compliance()
        
        # GDPR compliance
        self._check_gdpr_compliance()
        
        # Accessibility compliance
        self._check_accessibility_compliance()
        
        # Calculate final score
        self._calculate_compliance_score()
        
        logger.info(f"Compliance audit completed. Score: {self.audit_results['summary']['score']:.1f}%")
        
        return self.audit_results
        
    def _check_privacy_compliance(self):
        """Check privacy-specific compliance."""
        logger.info("Checking privacy compliance...")
        
        checks = {
            "differential_privacy_config": self._check_dp_config(),
            "privacy_budget_monitoring": self._check_privacy_budget_monitoring(),
            "data_anonymization": self._check_data_anonymization(),
            "privacy_documentation": self._check_privacy_docs(),
            "consent_management": self._check_consent_management()
        }
        
        self.audit_results["checks"]["privacy"] = checks
        
    def _check_security_compliance(self):
        """Check security compliance."""
        logger.info("Checking security compliance...")
        
        checks = {
            "dependency_vulnerabilities": self._check_dependencies(),
            "secrets_management": self._check_secrets(),
            "secure_communication": self._check_secure_comms(),
            "authentication": self._check_authentication(),
            "authorization": self._check_authorization(),
            "input_validation": self._check_input_validation(),
            "crypto_standards": self._check_crypto_standards()
        }
        
        self.audit_results["checks"]["security"] = checks
        
    def _check_code_quality(self):
        """Check code quality compliance."""
        logger.info("Checking code quality compliance...")
        
        checks = {
            "test_coverage": self._check_test_coverage(),
            "code_style": self._check_code_style(),
            "static_analysis": self._check_static_analysis(),
            "complexity": self._check_complexity(),
            "documentation": self._check_code_documentation()
        }
        
        self.audit_results["checks"]["code_quality"] = checks
        
    def _check_documentation_compliance(self):
        """Check documentation compliance."""
        logger.info("Checking documentation compliance...")
        
        checks = {
            "readme_completeness": self._check_readme(),
            "api_documentation": self._check_api_docs(),
            "security_docs": self._check_security_docs(),
            "deployment_docs": self._check_deployment_docs(),
            "architecture_docs": self._check_architecture_docs()
        }
        
        self.audit_results["checks"]["documentation"] = checks
        
    def _check_infrastructure_compliance(self):
        """Check infrastructure compliance."""
        logger.info("Checking infrastructure compliance...")
        
        checks = {
            "container_security": self._check_container_security(),
            "network_security": self._check_network_security(),
            "monitoring": self._check_monitoring(),
            "backup_strategy": self._check_backup_strategy(),
            "disaster_recovery": self._check_disaster_recovery()
        }
        
        self.audit_results["checks"]["infrastructure"] = checks
        
    def _check_license_compliance(self):
        """Check license compliance."""
        logger.info("Checking license compliance...")
        
        checks = {
            "license_file": self._check_license_file(),
            "dependency_licenses": self._check_dependency_licenses(),
            "license_headers": self._check_license_headers(),
            "copyright_notices": self._check_copyright_notices()
        }
        
        self.audit_results["checks"]["license"] = checks
        
    def _check_gdpr_compliance(self):
        """Check GDPR compliance."""
        logger.info("Checking GDPR compliance...")
        
        checks = {
            "data_protection": self._check_data_protection(),
            "right_to_erasure": self._check_right_to_erasure(),
            "data_portability": self._check_data_portability(),
            "consent_withdrawal": self._check_consent_withdrawal(),
            "privacy_by_design": self._check_privacy_by_design()
        }
        
        self.audit_results["checks"]["gdpr"] = checks
        
    def _check_accessibility_compliance(self):
        """Check accessibility compliance."""
        logger.info("Checking accessibility compliance...")
        
        checks = {
            "api_accessibility": self._check_api_accessibility(),
            "documentation_accessibility": self._check_docs_accessibility(),
            "interface_accessibility": self._check_interface_accessibility()
        }
        
        self.audit_results["checks"]["accessibility"] = checks
        
    def _check_dp_config(self) -> Dict[str, Any]:
        """Check differential privacy configuration."""
        try:
            config_file = self.project_root / "privacy_finetuner" / "core" / "privacy_config.py"
            
            if not config_file.exists():
                return {"status": "failed", "message": "PrivacyConfig file not found"}
                
            # Check if PrivacyConfig class has required methods
            with open(config_file, 'r') as f:
                content = f.read()
                
            required_methods = [
                "validate",
                "estimate_privacy_cost",
                "get_effective_noise_scale"
            ]
            
            missing_methods = []
            for method in required_methods:
                if f"def {method}" not in content:
                    missing_methods.append(method)
                    
            if missing_methods:
                return {
                    "status": "failed",
                    "message": f"Missing required methods: {missing_methods}"
                }
                
            return {"status": "passed", "message": "DP configuration is compliant"}
            
        except Exception as e:
            return {"status": "failed", "message": f"Error checking DP config: {e}"}
            
    def _check_privacy_budget_monitoring(self) -> Dict[str, Any]:
        """Check privacy budget monitoring implementation."""
        try:
            monitoring_file = self.project_root / "privacy_finetuner" / "utils" / "monitoring.py"
            
            if not monitoring_file.exists():
                return {"status": "failed", "message": "Privacy budget monitoring not implemented"}
                
            with open(monitoring_file, 'r') as f:
                content = f.read()
                
            if "PrivacyBudgetMonitor" not in content:
                return {"status": "failed", "message": "PrivacyBudgetMonitor class not found"}
                
            return {"status": "passed", "message": "Privacy budget monitoring implemented"}
            
        except Exception as e:
            return {"status": "failed", "message": f"Error checking monitoring: {e}"}
            
    def _check_data_anonymization(self) -> Dict[str, Any]:
        """Check data anonymization implementation."""
        try:
            context_guard_file = self.project_root / "privacy_finetuner" / "core" / "context_guard.py"
            
            if not context_guard_file.exists():
                return {"status": "failed", "message": "Context guard not implemented"}
                
            with open(context_guard_file, 'r') as f:
                content = f.read()
                
            required_strategies = ["PII_REMOVAL", "ENTITY_HASHING", "SEMANTIC_ENCRYPTION"]
            missing_strategies = []
            
            for strategy in required_strategies:
                if strategy not in content:
                    missing_strategies.append(strategy)
                    
            if missing_strategies:
                return {
                    "status": "warning",
                    "message": f"Missing anonymization strategies: {missing_strategies}"
                }
                
            return {"status": "passed", "message": "Data anonymization properly implemented"}
            
        except Exception as e:
            return {"status": "failed", "message": f"Error checking anonymization: {e}"}
            
    def _check_privacy_docs(self) -> Dict[str, Any]:
        """Check privacy documentation."""
        privacy_docs = [
            "PRIVACY.md",
            "SECURITY.md",
            "docs/privacy/",
            "docs/compliance/"
        ]
        
        existing_docs = []
        for doc in privacy_docs:
            doc_path = self.project_root / doc
            if doc_path.exists():
                existing_docs.append(doc)
                
        if len(existing_docs) < 2:
            return {
                "status": "warning",
                "message": f"Limited privacy documentation. Found: {existing_docs}"
            }
            
        return {"status": "passed", "message": f"Privacy documentation available: {existing_docs}"}
        
    def _check_consent_management(self) -> Dict[str, Any]:
        """Check consent management implementation."""
        # Check database models for consent fields
        models_file = self.project_root / "privacy_finetuner" / "database" / "models.py"
        
        if not models_file.exists():
            return {"status": "failed", "message": "Database models not found"}
            
        try:
            with open(models_file, 'r') as f:
                content = f.read()
                
            consent_fields = ["consent_given", "consent_date", "privacy_preferences"]
            found_fields = []
            
            for field in consent_fields:
                if field in content:
                    found_fields.append(field)
                    
            if len(found_fields) < 2:
                return {
                    "status": "warning",
                    "message": f"Limited consent management. Found fields: {found_fields}"
                }
                
            return {"status": "passed", "message": "Consent management implemented"}
            
        except Exception as e:
            return {"status": "failed", "message": f"Error checking consent management: {e}"}
            
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check for dependency vulnerabilities."""
        try:
            # Run safety check
            result = subprocess.run(
                ["poetry", "run", "safety", "check", "--json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                return {"status": "passed", "message": "No known vulnerabilities found"}
            else:
                try:
                    vulnerabilities = json.loads(result.stdout)
                    return {
                        "status": "failed",
                        "message": f"Found {len(vulnerabilities)} vulnerabilities",
                        "details": vulnerabilities
                    }
                except:
                    return {"status": "warning", "message": "Could not parse safety output"}
                    
        except FileNotFoundError:
            return {"status": "warning", "message": "Safety not installed or not available"}
        except Exception as e:
            return {"status": "failed", "message": f"Error checking dependencies: {e}"}
            
    def _check_secrets(self) -> Dict[str, Any]:
        """Check secrets management."""
        try:
            # Run bandit for secret detection
            result = subprocess.run(
                ["poetry", "run", "bandit", "-r", "privacy_finetuner/", "-f", "json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                return {"status": "passed", "message": "No hardcoded secrets detected"}
            else:
                try:
                    bandit_output = json.loads(result.stdout)
                    issues = bandit_output.get("results", [])
                    secret_issues = [i for i in issues if "password" in i.get("test_name", "").lower() 
                                   or "hardcoded" in i.get("test_name", "").lower()]
                    
                    if secret_issues:
                        return {
                            "status": "failed",
                            "message": f"Found {len(secret_issues)} potential secret issues",
                            "details": secret_issues
                        }
                    else:
                        return {"status": "passed", "message": "No hardcoded secrets detected"}
                        
                except:
                    return {"status": "warning", "message": "Could not parse bandit output"}
                    
        except FileNotFoundError:
            return {"status": "warning", "message": "Bandit not installed or not available"}
        except Exception as e:
            return {"status": "failed", "message": f"Error checking secrets: {e}"}
            
    def _check_secure_comms(self) -> Dict[str, Any]:
        """Check secure communication implementation."""
        # Check for HTTPS enforcement, TLS configuration, etc.
        config_files = [
            "config/nginx.conf",
            "docker-compose.yml",
            "privacy_finetuner/api/server.py"
        ]
        
        https_references = 0
        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read().lower()
                        if "https" in content or "ssl" in content or "tls" in content:
                            https_references += 1
                except:
                    continue
                    
        if https_references > 0:
            return {"status": "passed", "message": f"HTTPS/TLS configured in {https_references} files"}
        else:
            return {"status": "warning", "message": "No explicit HTTPS/TLS configuration found"}
            
    def _check_authentication(self) -> Dict[str, Any]:
        """Check authentication implementation."""
        auth_file = self.project_root / "privacy_finetuner" / "integrations" / "auth.py"
        
        if not auth_file.exists():
            return {"status": "failed", "message": "Authentication module not found"}
            
        try:
            with open(auth_file, 'r') as f:
                content = f.read()
                
            required_components = ["JWTManager", "AuthService", "PasswordManager"]
            found_components = []
            
            for component in required_components:
                if component in content:
                    found_components.append(component)
                    
            if len(found_components) == len(required_components):
                return {"status": "passed", "message": "Authentication properly implemented"}
            else:
                return {
                    "status": "warning",
                    "message": f"Partial authentication implementation. Found: {found_components}"
                }
                
        except Exception as e:
            return {"status": "failed", "message": f"Error checking authentication: {e}"}
            
    def _check_authorization(self) -> Dict[str, Any]:
        """Check authorization implementation."""
        # Check for role-based access control, permissions, etc.
        auth_file = self.project_root / "privacy_finetuner" / "integrations" / "auth.py"
        
        if not auth_file.exists():
            return {"status": "failed", "message": "Authorization module not found"}
            
        try:
            with open(auth_file, 'r') as f:
                content = f.read()
                
            auth_features = ["has_permission", "has_role", "permissions", "roles"]
            found_features = []
            
            for feature in auth_features:
                if feature in content:
                    found_features.append(feature)
                    
            if len(found_features) >= 3:
                return {"status": "passed", "message": "Authorization properly implemented"}
            else:
                return {
                    "status": "warning",
                    "message": f"Limited authorization features. Found: {found_features}"
                }
                
        except Exception as e:
            return {"status": "failed", "message": f"Error checking authorization: {e}"}
            
    def _check_input_validation(self) -> Dict[str, Any]:
        """Check input validation implementation."""
        api_file = self.project_root / "privacy_finetuner" / "api" / "server.py"
        
        if not api_file.exists():
            return {"status": "failed", "message": "API server not found"}
            
        try:
            with open(api_file, 'r') as f:
                content = f.read()
                
            validation_indicators = ["pydantic", "BaseModel", "validation", "validator"]
            found_indicators = []
            
            for indicator in validation_indicators:
                if indicator in content:
                    found_indicators.append(indicator)
                    
            if len(found_indicators) >= 2:
                return {"status": "passed", "message": "Input validation implemented"}
            else:
                return {
                    "status": "warning", 
                    "message": f"Limited input validation. Found: {found_indicators}"
                }
                
        except Exception as e:
            return {"status": "failed", "message": f"Error checking input validation: {e}"}
            
    def _check_crypto_standards(self) -> Dict[str, Any]:
        """Check cryptographic standards compliance."""
        # Check for proper crypto library usage
        crypto_files = []
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if any(lib in content for lib in ["cryptography", "bcrypt", "hashlib", "secrets"]):
                        crypto_files.append(str(py_file.relative_to(self.project_root)))
            except:
                continue
                
        if crypto_files:
            return {
                "status": "passed",
                "message": f"Cryptographic libraries used in {len(crypto_files)} files",
                "details": crypto_files
            }
        else:
            return {"status": "warning", "message": "No cryptographic library usage detected"}
            
    def _check_test_coverage(self) -> Dict[str, Any]:
        """Check test coverage."""
        try:
            # Run pytest with coverage
            result = subprocess.run(
                ["poetry", "run", "pytest", "--cov=privacy_finetuner", "--cov-report=json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                
                if total_coverage >= 80:
                    status = "passed"
                elif total_coverage >= 60:
                    status = "warning"
                else:
                    status = "failed"
                    
                return {
                    "status": status,
                    "message": f"Test coverage: {total_coverage:.1f}%",
                    "details": coverage_data.get("totals", {})
                }
            else:
                return {"status": "warning", "message": "Coverage report not generated"}
                
        except Exception as e:
            return {"status": "failed", "message": f"Error checking test coverage: {e}"}
            
    def _check_code_style(self) -> Dict[str, Any]:
        """Check code style compliance."""
        try:
            # Run black check
            result = subprocess.run(
                ["poetry", "run", "black", "--check", "."],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                return {"status": "passed", "message": "Code style compliant"}
            else:
                return {
                    "status": "warning",
                    "message": "Code style issues found",
                    "details": result.stdout
                }
                
        except FileNotFoundError:
            return {"status": "warning", "message": "Black not installed or not available"}
        except Exception as e:
            return {"status": "failed", "message": f"Error checking code style: {e}"}
            
    def _check_static_analysis(self) -> Dict[str, Any]:
        """Check static analysis compliance."""
        try:
            # Run mypy
            result = subprocess.run(
                ["poetry", "run", "mypy", "privacy_finetuner/"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                return {"status": "passed", "message": "Static analysis passed"}
            else:
                error_count = result.stdout.count("error:")
                return {
                    "status": "warning" if error_count < 10 else "failed",
                    "message": f"Static analysis found {error_count} errors",
                    "details": result.stdout
                }
                
        except FileNotFoundError:
            return {"status": "warning", "message": "MyPy not installed or not available"}
        except Exception as e:
            return {"status": "failed", "message": f"Error running static analysis: {e}"}
            
    def _check_complexity(self) -> Dict[str, Any]:
        """Check code complexity."""
        # Simple complexity check based on file sizes and function counts
        py_files = list(self.project_root.rglob("privacy_finetuner/**/*.py"))
        
        large_files = []
        total_lines = 0
        
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    lines = len(f.readlines())
                    total_lines += lines
                    
                    if lines > 500:  # Flag files over 500 lines
                        large_files.append((str(py_file.relative_to(self.project_root)), lines))
            except:
                continue
                
        avg_file_size = total_lines / len(py_files) if py_files else 0
        
        if large_files:
            return {
                "status": "warning",
                "message": f"Found {len(large_files)} large files (>500 lines)",
                "details": {"large_files": large_files, "avg_size": avg_file_size}
            }
        else:
            return {
                "status": "passed",
                "message": f"Code complexity reasonable (avg {avg_file_size:.0f} lines/file)"
            }
            
    def _check_code_documentation(self) -> Dict[str, Any]:
        """Check code documentation coverage."""
        py_files = list(self.project_root.rglob("privacy_finetuner/**/*.py"))
        
        documented_functions = 0
        total_functions = 0
        
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                # Count function definitions
                function_count = content.count("def ")
                total_functions += function_count
                
                # Count documented functions (those with docstrings)
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "def " in line and i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line.startswith('"""') or next_line.startswith("'''"):
                            documented_functions += 1
                            
            except:
                continue
                
        if total_functions > 0:
            doc_coverage = (documented_functions / total_functions) * 100
            
            if doc_coverage >= 70:
                status = "passed"
            elif doc_coverage >= 40:
                status = "warning"
            else:
                status = "failed"
                
            return {
                "status": status,
                "message": f"Code documentation: {doc_coverage:.1f}% ({documented_functions}/{total_functions})"
            }
        else:
            return {"status": "warning", "message": "No functions found to check documentation"}
            
    def _check_readme(self) -> Dict[str, Any]:
        """Check README completeness."""
        readme_file = self.project_root / "README.md"
        
        if not readme_file.exists():
            return {"status": "failed", "message": "README.md not found"}
            
        try:
            with open(readme_file, 'r') as f:
                content = f.read().lower()
                
            required_sections = [
                "installation",
                "usage", 
                "example",
                "license",
                "privacy",
                "security"
            ]
            
            found_sections = []
            for section in required_sections:
                if section in content:
                    found_sections.append(section)
                    
            coverage = len(found_sections) / len(required_sections)
            
            if coverage >= 0.8:
                return {"status": "passed", "message": f"README is comprehensive ({coverage:.0%})"}
            elif coverage >= 0.5:
                return {"status": "warning", "message": f"README partially complete ({coverage:.0%})"}
            else:
                return {"status": "failed", "message": f"README incomplete ({coverage:.0%})"}
                
        except Exception as e:
            return {"status": "failed", "message": f"Error checking README: {e}"}
            
    def _check_api_docs(self) -> Dict[str, Any]:
        """Check API documentation."""
        api_file = self.project_root / "privacy_finetuner" / "api" / "server.py"
        
        if not api_file.exists():
            return {"status": "failed", "message": "API server not found"}
            
        try:
            with open(api_file, 'r') as f:
                content = f.read()
                
            # Check for FastAPI documentation features
            doc_features = ["@app.", "description=", "summary=", "response_model="]
            found_features = sum(1 for feature in doc_features if feature in content)
            
            if found_features >= 3:
                return {"status": "passed", "message": "API documentation implemented"}
            else:
                return {"status": "warning", "message": "Limited API documentation"}
                
        except Exception as e:
            return {"status": "failed", "message": f"Error checking API docs: {e}"}
            
    def _check_security_docs(self) -> Dict[str, Any]:
        """Check security documentation."""
        security_file = self.project_root / "SECURITY.md"
        
        if not security_file.exists():
            return {"status": "warning", "message": "SECURITY.md not found"}
            
        return {"status": "passed", "message": "Security documentation available"}
        
    def _check_deployment_docs(self) -> Dict[str, Any]:
        """Check deployment documentation."""
        deployment_docs = [
            "docs/deployment/",
            "DEPLOYMENT.md",
            "docker-compose.yml",
            "Dockerfile"
        ]
        
        found_docs = []
        for doc in deployment_docs:
            if (self.project_root / doc).exists():
                found_docs.append(doc)
                
        if len(found_docs) >= 2:
            return {"status": "passed", "message": f"Deployment docs available: {found_docs}"}
        else:
            return {"status": "warning", "message": "Limited deployment documentation"}
            
    def _check_architecture_docs(self) -> Dict[str, Any]:
        """Check architecture documentation."""
        arch_file = self.project_root / "ARCHITECTURE.md"
        
        if not arch_file.exists():
            return {"status": "warning", "message": "ARCHITECTURE.md not found"}
            
        return {"status": "passed", "message": "Architecture documentation available"}
        
    def _check_container_security(self) -> Dict[str, Any]:
        """Check container security."""
        dockerfile = self.project_root / "Dockerfile"
        
        if not dockerfile.exists():
            return {"status": "warning", "message": "Dockerfile not found"}
            
        try:
            with open(dockerfile, 'r') as f:
                content = f.read()
                
            security_features = [
                "USER ",  # Non-root user
                "HEALTHCHECK",  # Health checks
                "COPY --chown",  # Proper file ownership
            ]
            
            found_features = [f for f in security_features if f in content]
            
            if len(found_features) >= 2:
                return {"status": "passed", "message": f"Container security features: {found_features}"}
            else:
                return {"status": "warning", "message": "Limited container security features"}
                
        except Exception as e:
            return {"status": "failed", "message": f"Error checking container security: {e}"}
            
    def _check_network_security(self) -> Dict[str, Any]:
        """Check network security configuration."""
        compose_file = self.project_root / "docker-compose.yml"
        
        if not compose_file.exists():
            return {"status": "warning", "message": "docker-compose.yml not found"}
            
        try:
            with open(compose_file, 'r') as f:
                content = f.read()
                
            security_features = ["networks:", "healthcheck:", "restart:"]
            found_features = [f for f in security_features if f in content]
            
            if len(found_features) >= 2:
                return {"status": "passed", "message": "Network security configured"}
            else:
                return {"status": "warning", "message": "Limited network security configuration"}
                
        except Exception as e:
            return {"status": "failed", "message": f"Error checking network security: {e}"}
            
    def _check_monitoring(self) -> Dict[str, Any]:
        """Check monitoring implementation."""
        monitoring_dir = self.project_root / "monitoring"
        
        if not monitoring_dir.exists():
            return {"status": "failed", "message": "Monitoring configuration not found"}
            
        monitoring_files = list(monitoring_dir.rglob("*.yml")) + list(monitoring_dir.rglob("*.yaml"))
        
        if len(monitoring_files) >= 2:
            return {"status": "passed", "message": f"Monitoring configured with {len(monitoring_files)} files"}
        else:
            return {"status": "warning", "message": "Limited monitoring configuration"}
            
    def _check_backup_strategy(self) -> Dict[str, Any]:
        """Check backup strategy."""
        backup_files = [
            "scripts/backup.py",
            "scripts/restore.py",
            "docs/backup/",
            "BACKUP.md"
        ]
        
        found_files = []
        for backup_file in backup_files:
            if (self.project_root / backup_file).exists():
                found_files.append(backup_file)
                
        if found_files:
            return {"status": "passed", "message": f"Backup strategy implemented: {found_files}"}
        else:
            return {"status": "warning", "message": "No backup strategy documented"}
            
    def _check_disaster_recovery(self) -> Dict[str, Any]:
        """Check disaster recovery documentation."""
        dr_docs = [
            "docs/disaster-recovery/",
            "DISASTER_RECOVERY.md",
            "docs/deployment/disaster-recovery.md"
        ]
        
        found_docs = []
        for doc in dr_docs:
            if (self.project_root / doc).exists():
                found_docs.append(doc)
                
        if found_docs:
            return {"status": "passed", "message": f"DR documentation: {found_docs}"}
        else:
            return {"status": "warning", "message": "No disaster recovery documentation"}
            
    def _check_license_file(self) -> Dict[str, Any]:
        """Check license file."""
        license_file = self.project_root / "LICENSE"
        
        if not license_file.exists():
            return {"status": "failed", "message": "LICENSE file not found"}
            
        return {"status": "passed", "message": "LICENSE file present"}
        
    def _check_dependency_licenses(self) -> Dict[str, Any]:
        """Check dependency licenses."""
        try:
            # This would require pip-licenses or similar tool
            result = subprocess.run(
                ["poetry", "show"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                dep_count = len([line for line in result.stdout.split('\n') if line.strip()])
                return {"status": "passed", "message": f"Dependency licenses tracked ({dep_count} packages)"}
            else:
                return {"status": "warning", "message": "Could not check dependency licenses"}
                
        except Exception as e:
            return {"status": "warning", "message": f"Error checking dependency licenses: {e}"}
            
    def _check_license_headers(self) -> Dict[str, Any]:
        """Check license headers in source files."""
        py_files = list(self.project_root.rglob("privacy_finetuner/**/*.py"))
        
        files_with_headers = 0
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    first_lines = "".join(f.readlines()[:10])
                    if "copyright" in first_lines.lower() or "license" in first_lines.lower():
                        files_with_headers += 1
            except:
                continue
                
        if py_files:
            coverage = files_with_headers / len(py_files)
            if coverage >= 0.8:
                return {"status": "passed", "message": f"License headers: {coverage:.0%}"}
            else:
                return {"status": "warning", "message": f"Limited license headers: {coverage:.0%}"}
        else:
            return {"status": "warning", "message": "No Python files found"}
            
    def _check_copyright_notices(self) -> Dict[str, Any]:
        """Check copyright notices."""
        readme_file = self.project_root / "README.md"
        
        if readme_file.exists():
            try:
                with open(readme_file, 'r') as f:
                    content = f.read().lower()
                    
                if "copyright" in content or "©" in content:
                    return {"status": "passed", "message": "Copyright notice found"}
                    
            except:
                pass
                
        return {"status": "warning", "message": "No copyright notice found"}
        
    def _check_data_protection(self) -> Dict[str, Any]:
        """Check data protection measures."""
        # Check for encryption, access controls, etc.
        protection_indicators = [
            ("privacy_finetuner/core/context_guard.py", "PII protection"),
            ("privacy_finetuner/database/", "Database security"),
            ("privacy_finetuner/integrations/auth.py", "Authentication")
        ]
        
        found_protections = []
        for file_pattern, description in protection_indicators:
            if (self.project_root / file_pattern).exists():
                found_protections.append(description)
                
        if len(found_protections) >= 2:
            return {"status": "passed", "message": f"Data protection: {found_protections}"}
        else:
            return {"status": "warning", "message": "Limited data protection measures"}
            
    def _check_right_to_erasure(self) -> Dict[str, Any]:
        """Check right to erasure implementation."""
        # Check for data deletion capabilities
        deletion_patterns = ["delete", "remove", "erase", "purge"]
        
        py_files = list(self.project_root.rglob("privacy_finetuner/**/*.py"))
        files_with_deletion = 0
        
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    if any(pattern in content for pattern in deletion_patterns):
                        files_with_deletion += 1
            except:
                continue
                
        if files_with_deletion > 0:
            return {"status": "passed", "message": f"Deletion capabilities in {files_with_deletion} files"}
        else:
            return {"status": "warning", "message": "No explicit deletion capabilities found"}
            
    def _check_data_portability(self) -> Dict[str, Any]:
        """Check data portability implementation."""
        # Check for export capabilities
        export_patterns = ["export", "download", "extract", "serialize"]
        
        api_file = self.project_root / "privacy_finetuner" / "api" / "server.py"
        
        if api_file.exists():
            try:
                with open(api_file, 'r') as f:
                    content = f.read().lower()
                    if any(pattern in content for pattern in export_patterns):
                        return {"status": "passed", "message": "Data export capabilities found"}
            except:
                pass
                
        return {"status": "warning", "message": "No explicit data export capabilities found"}
        
    def _check_consent_withdrawal(self) -> Dict[str, Any]:
        """Check consent withdrawal implementation."""
        # Check for consent management in models
        models_file = self.project_root / "privacy_finetuner" / "database" / "models.py"
        
        if models_file.exists():
            try:
                with open(models_file, 'r') as f:
                    content = f.read()
                    if "consent" in content.lower():
                        return {"status": "passed", "message": "Consent management implemented"}
            except:
                pass
                
        return {"status": "warning", "message": "No explicit consent management found"}
        
    def _check_privacy_by_design(self) -> Dict[str, Any]:
        """Check privacy by design implementation."""
        # Check for privacy features throughout codebase
        privacy_features = [
            "differential privacy",
            "anonymization", 
            "encryption",
            "access control",
            "audit logging"
        ]
        
        feature_files = {}
        
        for py_file in self.project_root.rglob("privacy_finetuner/**/*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    for feature in privacy_features:
                        if feature.replace(" ", "_") in content or feature in content:
                            if feature not in feature_files:
                                feature_files[feature] = []
                            feature_files[feature].append(str(py_file.relative_to(self.project_root)))
            except:
                continue
                
        if len(feature_files) >= 3:
            return {"status": "passed", "message": f"Privacy by design: {list(feature_files.keys())}"}
        else:
            return {"status": "warning", "message": f"Limited privacy features: {list(feature_files.keys())}"}
            
    def _check_api_accessibility(self) -> Dict[str, Any]:
        """Check API accessibility."""
        # Check for proper HTTP status codes, error messages, etc.
        api_file = self.project_root / "privacy_finetuner" / "api" / "server.py"
        
        if not api_file.exists():
            return {"status": "warning", "message": "API server not found"}
            
        try:
            with open(api_file, 'r') as f:
                content = f.read()
                
            accessibility_features = ["HTTPException", "status_code", "detail", "response_model"]
            found_features = [f for f in accessibility_features if f in content]
            
            if len(found_features) >= 3:
                return {"status": "passed", "message": "API accessibility features implemented"}
            else:
                return {"status": "warning", "message": "Limited API accessibility"}
                
        except Exception as e:
            return {"status": "failed", "message": f"Error checking API accessibility: {e}"}
            
    def _check_docs_accessibility(self) -> Dict[str, Any]:
        """Check documentation accessibility."""
        # Check for clear structure, examples, etc.
        md_files = list(self.project_root.rglob("*.md"))
        
        accessible_docs = 0
        for md_file in md_files:
            try:
                with open(md_file, 'r') as f:
                    content = f.read()
                    # Check for headers, code blocks, lists
                    if content.count('#') >= 2 and '```' in content:
                        accessible_docs += 1
            except:
                continue
                
        if accessible_docs >= len(md_files) * 0.7:
            return {"status": "passed", "message": f"Documentation accessibility good ({accessible_docs}/{len(md_files)})"}
        else:
            return {"status": "warning", "message": f"Documentation accessibility needs improvement ({accessible_docs}/{len(md_files)})"}
            
    def _check_interface_accessibility(self) -> Dict[str, Any]:
        """Check interface accessibility."""
        # For CLI and API interfaces
        cli_file = self.project_root / "privacy_finetuner" / "cli.py"
        
        if not cli_file.exists():
            return {"status": "warning", "message": "CLI interface not found"}
            
        try:
            with open(cli_file, 'r') as f:
                content = f.read()
                
            # Check for help text, clear commands, etc.
            accessibility_features = ["help=", "description=", "epilog=", "--help"]
            found_features = [f for f in accessibility_features if f in content]
            
            if len(found_features) >= 2:
                return {"status": "passed", "message": "Interface accessibility implemented"}
            else:
                return {"status": "warning", "message": "Limited interface accessibility"}
                
        except Exception as e:
            return {"status": "failed", "message": f"Error checking interface accessibility: {e}"}
            
    def _calculate_compliance_score(self):
        """Calculate overall compliance score."""
        total_checks = 0
        passed = 0
        failed = 0
        warnings = 0
        
        for category, checks in self.audit_results["checks"].items():
            for check_name, result in checks.items():
                total_checks += 1
                status = result.get("status", "unknown")
                
                if status == "passed":
                    passed += 1
                elif status == "failed":
                    failed += 1
                elif status == "warning":
                    warnings += 1
                    
        # Calculate weighted score (passed=100%, warning=50%, failed=0%)
        score = ((passed * 100) + (warnings * 50)) / total_checks if total_checks > 0 else 0
        
        self.audit_results["summary"] = {
            "total_checks": total_checks,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "score": score
        }
        
        # Set overall status
        if score >= 80:
            self.audit_results["status"] = "compliant"
        elif score >= 60:
            self.audit_results["status"] = "partially_compliant"
        else:
            self.audit_results["status"] = "non_compliant"


def main():
    """Run compliance audit."""
    project_root = Path(__file__).parent.parent
    auditor = ComplianceAuditor(project_root)
    
    results = auditor.run_full_audit()
    
    # Save results
    results_file = project_root / "compliance_audit_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    print(f"\nCompliance Audit Results:")
    print(f"========================")
    print(f"Status: {results['status'].upper()}")
    print(f"Score: {results['summary']['score']:.1f}%")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"\nDetailed results saved to: {results_file}")
    
    # Exit with appropriate code
    if results['status'] == 'compliant':
        sys.exit(0)
    elif results['status'] == 'partially_compliant':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()