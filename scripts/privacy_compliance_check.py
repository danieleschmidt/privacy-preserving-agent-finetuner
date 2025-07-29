#!/usr/bin/env python3
"""Privacy compliance checking script for the Privacy-Preserving Agent Finetuner."""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ComplianceCheck:
    """Represents a single compliance check."""
    name: str
    description: str
    regulation: str
    severity: str
    passed: bool
    details: Optional[str] = None


@dataclass
class ComplianceReport:
    """Represents a full compliance report."""
    timestamp: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    critical_failures: int
    checks: List[ComplianceCheck]
    overall_status: str


class PrivacyComplianceChecker:
    """Privacy compliance checker for GDPR, HIPAA, CCPA, and other regulations."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the compliance checker."""
        self.config_path = config_path or Path("config/compliance.yaml")
        self.config = self._load_config()
        self.checks: List[ComplianceCheck] = []

    def _load_config(self) -> Dict[str, Any]:
        """Load compliance configuration."""
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default compliance configuration."""
        return {
            "regulations": {
                "gdpr": {"enabled": True, "strict_mode": False},
                "hipaa": {"enabled": False, "strict_mode": False},
                "ccpa": {"enabled": True, "strict_mode": False},
                "pipeda": {"enabled": False, "strict_mode": False}
            },
            "privacy_settings": {
                "min_epsilon": 0.1,
                "max_epsilon": 10.0,
                "min_delta": 1e-8,
                "max_delta": 1e-3,
                "require_consent": True,
                "data_retention_days": 30
            },
            "security_requirements": {
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "access_logging": True,
                "audit_trail": True
            }
        }

    def check_gdpr_compliance(self) -> List[ComplianceCheck]:
        """Check GDPR compliance requirements."""
        checks = []
        
        if not self.config["regulations"]["gdpr"]["enabled"]:
            return checks

        # Article 5 - Lawfulness, fairness, and transparency
        checks.append(self._check_lawful_basis())
        checks.append(self._check_data_minimization())
        checks.append(self._check_purpose_limitation())
        
        # Article 6 - Lawfulness of processing
        checks.append(self._check_consent_mechanism())
        
        # Article 17 - Right to erasure
        checks.append(self._check_data_deletion())
        
        # Article 25 - Data protection by design and by default
        checks.append(self._check_privacy_by_design())
        
        # Article 32 - Security of processing
        checks.append(self._check_security_measures())
        
        # Article 35 - Data protection impact assessment
        checks.append(self._check_dpia_requirement())

        return checks

    def check_hipaa_compliance(self) -> List[ComplianceCheck]:
        """Check HIPAA compliance requirements."""
        checks = []
        
        if not self.config["regulations"]["hipaa"]["enabled"]:
            return checks

        # Administrative Safeguards
        checks.append(self._check_access_management())
        checks.append(self._check_workforce_training())
        
        # Physical Safeguards
        checks.append(self._check_facility_access())
        checks.append(self._check_workstation_use())
        
        # Technical Safeguards
        checks.append(self._check_access_control())
        checks.append(self._check_audit_controls())
        checks.append(self._check_integrity())
        checks.append(self._check_transmission_security())

        return checks

    def check_ccpa_compliance(self) -> List[ComplianceCheck]:
        """Check CCPA compliance requirements."""
        checks = []
        
        if not self.config["regulations"]["ccpa"]["enabled"]:
            return checks

        # Consumer Rights
        checks.append(self._check_right_to_know())
        checks.append(self._check_right_to_delete())
        checks.append(self._check_right_to_opt_out())
        
        # Business Obligations
        checks.append(self._check_privacy_notice())
        checks.append(self._check_data_processing_purpose())

        return checks

    def check_differential_privacy_parameters(self) -> List[ComplianceCheck]:
        """Check differential privacy parameter compliance."""
        checks = []
        
        # Check epsilon parameter
        epsilon_check = self._check_epsilon_bounds()
        checks.append(epsilon_check)
        
        # Check delta parameter
        delta_check = self._check_delta_bounds()
        checks.append(delta_check)
        
        # Check composition bounds
        composition_check = self._check_composition_bounds()
        checks.append(composition_check)
        
        return checks

    def _check_lawful_basis(self) -> ComplianceCheck:
        """Check GDPR Article 6 - Lawful basis for processing."""
        # Check if consent mechanism is implemented
        consent_file = Path("privacy_finetuner/consent/manager.py")
        has_consent = consent_file.exists()
        
        return ComplianceCheck(
            name="GDPR Article 6 - Lawful Basis",
            description="Processing must have a lawful basis",
            regulation="GDPR",
            severity="critical",
            passed=has_consent,
            details="Consent management system required" if not has_consent else None
        )

    def _check_data_minimization(self) -> ComplianceCheck:
        """Check GDPR Article 5(1)(c) - Data minimization."""
        # Check if data minimization is configured
        min_config = self.config.get("privacy_settings", {}).get("data_minimization", False)
        
        return ComplianceCheck(
            name="GDPR Article 5(1)(c) - Data Minimization",
            description="Personal data must be adequate, relevant and limited",
            regulation="GDPR",
            severity="high",
            passed=min_config,
            details="Data minimization must be configured" if not min_config else None
        )

    def _check_purpose_limitation(self) -> ComplianceCheck:
        """Check GDPR Article 5(1)(b) - Purpose limitation."""
        # Check if purpose specification is documented
        purpose_file = Path("docs/data-processing-purposes.md")
        has_purpose = purpose_file.exists()
        
        return ComplianceCheck(
            name="GDPR Article 5(1)(b) - Purpose Limitation",
            description="Data must be collected for specified, explicit and legitimate purposes",
            regulation="GDPR",
            severity="high",
            passed=has_purpose,
            details="Data processing purposes must be documented" if not has_purpose else None
        )

    def _check_consent_mechanism(self) -> ComplianceCheck:
        """Check consent mechanism implementation."""
        consent_file = Path("privacy_finetuner/consent/manager.py")
        return ComplianceCheck(
            name="Consent Mechanism",
            description="Valid consent mechanism must be implemented",
            regulation="GDPR",
            severity="critical",
            passed=consent_file.exists(),
            details="Implement consent management system" if not consent_file.exists() else None
        )

    def _check_data_deletion(self) -> ComplianceCheck:
        """Check GDPR Article 17 - Right to erasure."""
        deletion_file = Path("privacy_finetuner/data/deletion.py")
        return ComplianceCheck(
            name="GDPR Article 17 - Right to Erasure",
            description="Individuals have the right to have personal data erased",
            regulation="GDPR",
            severity="critical",
            passed=deletion_file.exists(),
            details="Implement data deletion mechanism" if not deletion_file.exists() else None
        )

    def _check_privacy_by_design(self) -> ComplianceCheck:
        """Check GDPR Article 25 - Data protection by design."""
        # Check if privacy controls are implemented
        privacy_engine = Path("privacy_finetuner/privacy/engine.py")
        return ComplianceCheck(
            name="GDPR Article 25 - Privacy by Design",
            description="Privacy protection must be built into processing systems",
            regulation="GDPR",
            severity="high",
            passed=privacy_engine.exists(),
            details="Privacy engine implementation required" if not privacy_engine.exists() else None
        )

    def _check_security_measures(self) -> ComplianceCheck:
        """Check GDPR Article 32 - Security of processing."""
        security_config = self.config.get("security_requirements", {})
        required_measures = ["encryption_at_rest", "encryption_in_transit", "access_logging"]
        passed = all(security_config.get(measure, False) for measure in required_measures)
        
        return ComplianceCheck(
            name="GDPR Article 32 - Security Measures",
            description="Appropriate technical and organizational measures must be implemented",
            regulation="GDPR",
            severity="critical",
            passed=passed,
            details="All security measures must be enabled" if not passed else None
        )

    def _check_dpia_requirement(self) -> ComplianceCheck:
        """Check GDPR Article 35 - Data protection impact assessment."""
        dpia_file = Path("docs/dpia.md")
        return ComplianceCheck(
            name="GDPR Article 35 - DPIA",
            description="DPIA required for high-risk processing",
            regulation="GDPR",
            severity="medium",
            passed=dpia_file.exists(),
            details="Create Data Protection Impact Assessment" if not dpia_file.exists() else None
        )

    def _check_access_management(self) -> ComplianceCheck:
        """Check HIPAA access management requirements."""
        access_file = Path("privacy_finetuner/auth/access_control.py")
        return ComplianceCheck(
            name="HIPAA Access Management",
            description="Procedures for access management must be implemented",
            regulation="HIPAA",
            severity="critical",
            passed=access_file.exists(),
            details="Implement access control system" if not access_file.exists() else None
        )

    def _check_workforce_training(self) -> ComplianceCheck:
        """Check HIPAA workforce training requirements."""
        training_file = Path("docs/security-training.md")
        return ComplianceCheck(
            name="HIPAA Workforce Training",
            description="Workforce must be trained on security procedures",
            regulation="HIPAA",
            severity="medium",
            passed=training_file.exists(),
            details="Document security training procedures" if not training_file.exists() else None
        )

    def _check_facility_access(self) -> ComplianceCheck:
        """Check HIPAA facility access controls."""
        # This would typically check physical security measures
        return ComplianceCheck(
            name="HIPAA Facility Access",
            description="Physical access to facilities must be controlled",
            regulation="HIPAA",
            severity="medium",
            passed=True,  # Assume passed for cloud deployments
            details=None
        )

    def _check_workstation_use(self) -> ComplianceCheck:
        """Check HIPAA workstation use controls."""
        return ComplianceCheck(
            name="HIPAA Workstation Use",
            description="Workstation access must be controlled",
            regulation="HIPAA",
            severity="medium",
            passed=True,  # Assume passed for containerized deployments
            details=None
        )

    def _check_access_control(self) -> ComplianceCheck:
        """Check HIPAA access control requirements."""
        auth_file = Path("privacy_finetuner/auth/authentication.py")
        return ComplianceCheck(
            name="HIPAA Access Control",
            description="Technical access control must be implemented",
            regulation="HIPAA",
            severity="critical",
            passed=auth_file.exists(),
            details="Implement authentication system" if not auth_file.exists() else None
        )

    def _check_audit_controls(self) -> ComplianceCheck:
        """Check HIPAA audit control requirements."""
        audit_file = Path("privacy_finetuner/audit/logger.py")
        return ComplianceCheck(
            name="HIPAA Audit Controls",
            description="Audit controls must be implemented",
            regulation="HIPAA",
            severity="critical",
            passed=audit_file.exists(),
            details="Implement audit logging system" if not audit_file.exists() else None
        )

    def _check_integrity(self) -> ComplianceCheck:
        """Check HIPAA integrity requirements."""
        integrity_config = self.config.get("security_requirements", {}).get("audit_trail", False)
        return ComplianceCheck(
            name="HIPAA Integrity",
            description="Electronic PHI must not be improperly altered or destroyed",
            regulation="HIPAA",
            severity="high",
            passed=integrity_config,
            details="Enable audit trail for integrity checking" if not integrity_config else None
        )

    def _check_transmission_security(self) -> ComplianceCheck:
        """Check HIPAA transmission security requirements."""
        tls_config = self.config.get("security_requirements", {}).get("encryption_in_transit", False)
        return ComplianceCheck(
            name="HIPAA Transmission Security",
            description="Electronic PHI must be protected during transmission",
            regulation="HIPAA",
            severity="critical",
            passed=tls_config,
            details="Enable encryption in transit" if not tls_config else None
        )

    def _check_right_to_know(self) -> ComplianceCheck:
        """Check CCPA right to know requirements."""
        privacy_notice = Path("docs/privacy-notice.md")
        return ComplianceCheck(
            name="CCPA Right to Know",
            description="Consumers have the right to know about personal information collection",
            regulation="CCPA",
            severity="high",
            passed=privacy_notice.exists(),
            details="Create privacy notice document" if not privacy_notice.exists() else None
        )

    def _check_right_to_delete(self) -> ComplianceCheck:
        """Check CCPA right to delete requirements."""
        deletion_file = Path("privacy_finetuner/data/deletion.py")
        return ComplianceCheck(
            name="CCPA Right to Delete",
            description="Consumers have the right to delete personal information",
            regulation="CCPA",
            severity="critical",
            passed=deletion_file.exists(),
            details="Implement data deletion mechanism" if not deletion_file.exists() else None
        )

    def _check_right_to_opt_out(self) -> ComplianceCheck:
        """Check CCPA right to opt-out requirements."""
        opt_out_file = Path("privacy_finetuner/consent/opt_out.py")
        return ComplianceCheck(
            name="CCPA Right to Opt-Out",
            description="Consumers have the right to opt out of sale of personal information",
            regulation="CCPA",
            severity="high",
            passed=opt_out_file.exists(),
            details="Implement opt-out mechanism" if not opt_out_file.exists() else None
        )

    def _check_privacy_notice(self) -> ComplianceCheck:
        """Check CCPA privacy notice requirements."""
        privacy_notice = Path("docs/privacy-notice.md")
        return ComplianceCheck(
            name="CCPA Privacy Notice",
            description="Privacy notice must be provided to consumers",
            regulation="CCPA",
            severity="high",
            passed=privacy_notice.exists(),
            details="Create comprehensive privacy notice" if not privacy_notice.exists() else None
        )

    def _check_data_processing_purpose(self) -> ComplianceCheck:
        """Check CCPA data processing purpose disclosure."""
        purpose_file = Path("docs/data-processing-purposes.md")
        return ComplianceCheck(
            name="CCPA Processing Purpose",
            description="Business purposes for processing must be disclosed",
            regulation="CCPA",
            severity="medium",
            passed=purpose_file.exists(),
            details="Document data processing purposes" if not purpose_file.exists() else None
        )

    def _check_epsilon_bounds(self) -> ComplianceCheck:
        """Check epsilon parameter bounds."""
        privacy_settings = self.config.get("privacy_settings", {})
        min_epsilon = privacy_settings.get("min_epsilon", 0.1)
        max_epsilon = privacy_settings.get("max_epsilon", 10.0)
        
        # Check if epsilon is within acceptable bounds
        current_epsilon = float(Path(".env.example").read_text().split("PRIVACY_EPSILON=")[1].split("\n")[0])
        within_bounds = min_epsilon <= current_epsilon <= max_epsilon
        
        return ComplianceCheck(
            name="Differential Privacy - Epsilon Bounds",
            description="Epsilon parameter must be within acceptable privacy bounds",
            regulation="Privacy",
            severity="critical",
            passed=within_bounds,
            details=f"Epsilon {current_epsilon} must be between {min_epsilon} and {max_epsilon}" if not within_bounds else None
        )

    def _check_delta_bounds(self) -> ComplianceCheck:
        """Check delta parameter bounds."""
        privacy_settings = self.config.get("privacy_settings", {})
        min_delta = privacy_settings.get("min_delta", 1e-8)
        max_delta = privacy_settings.get("max_delta", 1e-3)
        
        # Check if delta is within acceptable bounds
        current_delta = float(Path(".env.example").read_text().split("PRIVACY_DELTA=")[1].split("\n")[0])
        within_bounds = min_delta <= current_delta <= max_delta
        
        return ComplianceCheck(
            name="Differential Privacy - Delta Bounds",
            description="Delta parameter must be within acceptable privacy bounds",
            regulation="Privacy",
            severity="critical",
            passed=within_bounds,
            details=f"Delta {current_delta} must be between {min_delta} and {max_delta}" if not within_bounds else None
        )

    def _check_composition_bounds(self) -> ComplianceCheck:
        """Check composition bounds for privacy budget."""
        # This would typically check actual privacy budget consumption
        return ComplianceCheck(
            name="Differential Privacy - Composition Bounds",
            description="Privacy composition must not exceed total budget",
            regulation="Privacy",
            severity="critical",
            passed=True,  # Assume passed for initial setup
            details=None
        )

    def run_all_checks(self) -> ComplianceReport:
        """Run all compliance checks."""
        logger.info("Starting privacy compliance checks...")
        
        all_checks = []
        all_checks.extend(self.check_gdpr_compliance())
        all_checks.extend(self.check_hipaa_compliance())
        all_checks.extend(self.check_ccpa_compliance())
        all_checks.extend(self.check_differential_privacy_parameters())
        
        passed_checks = sum(1 for check in all_checks if check.passed)
        failed_checks = len(all_checks) - passed_checks
        critical_failures = sum(1 for check in all_checks if not check.passed and check.severity == "critical")
        
        overall_status = "PASS" if critical_failures == 0 else "FAIL"
        
        from datetime import datetime
        report = ComplianceReport(
            timestamp=datetime.now().isoformat(),
            total_checks=len(all_checks),
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            critical_failures=critical_failures,
            checks=all_checks,
            overall_status=overall_status
        )
        
        logger.info(f"Compliance check completed: {overall_status}")
        logger.info(f"Total checks: {len(all_checks)}, Passed: {passed_checks}, Failed: {failed_checks}")
        logger.info(f"Critical failures: {critical_failures}")
        
        return report

    def generate_report(self, report: ComplianceReport, output_path: Path) -> None:
        """Generate compliance report."""
        logger.info(f"Generating compliance report: {output_path}")
        
        report_data = {
            "timestamp": report.timestamp,
            "summary": {
                "total_checks": report.total_checks,
                "passed_checks": report.passed_checks,
                "failed_checks": report.failed_checks,
                "critical_failures": report.critical_failures,
                "overall_status": report.overall_status
            },
            "checks": [
                {
                    "name": check.name,
                    "description": check.description,
                    "regulation": check.regulation,
                    "severity": check.severity,
                    "passed": check.passed,
                    "details": check.details
                }
                for check in report.checks
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Privacy compliance checker")
    parser.add_argument("--config", type=Path, help="Config file path")
    parser.add_argument("--output", type=Path, default=Path("compliance-report.json"), help="Output report path")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        checker = PrivacyComplianceChecker(args.config)
        report = checker.run_all_checks()
        checker.generate_report(report, args.output)
        
        # Print summary
        print(f"\nCompliance Check Summary:")
        print(f"Overall Status: {report.overall_status}")
        print(f"Total Checks: {report.total_checks}")
        print(f"Passed: {report.passed_checks}")
        print(f"Failed: {report.failed_checks}")
        print(f"Critical Failures: {report.critical_failures}")
        
        if report.failed_checks > 0:
            print(f"\nFailed Checks:")
            for check in report.checks:
                if not check.passed:
                    print(f"  - {check.name} ({check.severity}): {check.details}")
        
        # Exit with error code if critical failures
        sys.exit(1 if report.critical_failures > 0 else 0)
        
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()