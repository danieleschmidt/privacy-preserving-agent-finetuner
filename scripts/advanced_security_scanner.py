#!/usr/bin/env python3
"""Advanced security scanner for privacy-preserving AI systems."""

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SecurityFinding:
    """Represents a security finding."""
    category: str
    severity: str
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cve_id: Optional[str] = None
    remediation: Optional[str] = None


@dataclass
class SecurityReport:
    """Represents a complete security report."""
    timestamp: str
    scan_type: str
    findings: List[SecurityFinding]
    summary: Dict[str, int]
    compliance_status: str


class AdvancedSecurityScanner:
    """Advanced security scanner with multiple detection engines."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the security scanner."""
        self.config_path = config_path or Path("config/security.yaml")
        self.config = self._load_config()
        self.findings: List[SecurityFinding] = []

    def _load_config(self) -> Dict[str, Any]:
        """Load security scanner configuration."""
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default security configuration."""
        return {
            "scanners": {
                "bandit": {"enabled": True, "severity": "medium"},
                "safety": {"enabled": True, "api_key": None},
                "semgrep": {"enabled": True, "rules": ["python", "security"]},
                "secrets": {"enabled": True, "entropy_threshold": 4.5},
                "dependencies": {"enabled": True, "sources": ["pypi", "npm"]},
                "container": {"enabled": True, "registry_scan": True},
                "infrastructure": {"enabled": True, "terraform_scan": True}
            },
            "thresholds": {
                "critical": 0,
                "high": 5,
                "medium": 20,
                "low": 50
            },
            "exclusions": {
                "files": ["tests/**", "docs/**"],
                "directories": [".git", "__pycache__"],
                "rules": []
            },
            "reporting": {
                "formats": ["json", "sarif", "html"],
                "output_dir": "security-reports"
            }
        }

    def run_bandit_scan(self) -> List[SecurityFinding]:
        """Run Bandit static analysis scanner."""
        findings = []
        
        if not self.config["scanners"]["bandit"]["enabled"]:
            return findings

        logger.info("Running Bandit security scan...")
        
        try:
            cmd = [
                "bandit", "-r", "privacy_finetuner/",
                "-f", "json",
                "--severity-level", self.config["scanners"]["bandit"]["severity"],
                "--confidence-level", "medium"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
                data = json.loads(result.stdout)
                
                for issue in data.get("results", []):
                    finding = SecurityFinding(
                        category="static_analysis",
                        severity=issue["issue_severity"].lower(),
                        title=issue["test_name"],
                        description=issue["issue_text"],
                        file_path=issue["filename"],
                        line_number=issue["line_number"],
                        remediation=issue.get("more_info", "")
                    )
                    findings.append(finding)
                    
        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
            findings.append(SecurityFinding(
                category="scanner_error",
                severity="medium",
                title="Bandit Scanner Error",
                description=f"Failed to run Bandit scanner: {e}"
            ))
        
        return findings

    def run_safety_scan(self) -> List[SecurityFinding]:
        """Run Safety vulnerability scanner."""
        findings = []
        
        if not self.config["scanners"]["safety"]["enabled"]:
            return findings

        logger.info("Running Safety dependency scan...")
        
        try:
            cmd = ["safety", "check", "--json", "--full-report"]
            
            if self.config["scanners"]["safety"].get("api_key"):
                cmd.extend(["--key", self.config["scanners"]["safety"]["api_key"]])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode in [0, 64]:  # 0 = safe, 64 = vulnerabilities found
                data = json.loads(result.stdout)
                
                for vuln in data.get("vulnerabilities", []):
                    finding = SecurityFinding(
                        category="dependency_vulnerability",
                        severity="high" if vuln.get("v") else "medium",
                        title=f"Vulnerable dependency: {vuln['package_name']}",
                        description=vuln["advisory"],
                        cve_id=vuln.get("cve"),
                        remediation=f"Update to version {vuln.get('safe_versions', 'latest')}"
                    )
                    findings.append(finding)
                    
        except Exception as e:
            logger.error(f"Safety scan failed: {e}")
            findings.append(SecurityFinding(
                category="scanner_error",
                severity="medium",
                title="Safety Scanner Error",
                description=f"Failed to run Safety scanner: {e}"
            ))
        
        return findings

    def run_secrets_scan(self) -> List[SecurityFinding]:
        """Run secrets detection scan."""
        findings = []
        
        if not self.config["scanners"]["secrets"]["enabled"]:
            return findings

        logger.info("Running secrets detection scan...")
        
        try:
            # Use detect-secrets
            cmd = ["detect-secrets", "scan", "--all-files", "--baseline", ".secrets.baseline"]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 1:  # Secrets found
                data = json.loads(result.stdout)
                
                for file_path, secrets in data.get("results", {}).items():
                    for secret in secrets:
                        finding = SecurityFinding(
                            category="secret_detection",
                            severity="critical",
                            title=f"Potential secret detected: {secret['type']}",
                            description=f"Possible {secret['type']} found in code",
                            file_path=file_path,
                            line_number=secret.get("line_number"),
                            remediation="Remove secret and use environment variables or secret management"
                        )
                        findings.append(finding)
                        
        except Exception as e:
            logger.error(f"Secrets scan failed: {e}")
            findings.append(SecurityFinding(
                category="scanner_error",
                severity="medium",
                title="Secrets Scanner Error",
                description=f"Failed to run secrets scanner: {e}"
            ))
        
        return findings

    def run_container_scan(self) -> List[SecurityFinding]:
        """Run container security scan."""
        findings = []
        
        if not self.config["scanners"]["container"]["enabled"]:
            return findings

        logger.info("Running container security scan...")
        
        try:
            # Scan Dockerfile
            cmd = ["hadolint", "--format", "json", "Dockerfile"]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                data = json.loads(result.stdout)
                
                for issue in data:
                    severity_map = {
                        "error": "high",
                        "warning": "medium", 
                        "info": "low",
                        "style": "low"
                    }
                    
                    finding = SecurityFinding(
                        category="container_security",
                        severity=severity_map.get(issue["level"], "medium"),
                        title=f"Docker issue: {issue['code']}",
                        description=issue["message"],
                        file_path="Dockerfile",
                        line_number=issue.get("line"),
                        remediation="Follow Docker security best practices"
                    )
                    findings.append(finding)
                    
        except Exception as e:
            logger.error(f"Container scan failed: {e}")
            findings.append(SecurityFinding(
                category="scanner_error",
                severity="medium",
                title="Container Scanner Error",
                description=f"Failed to run container scanner: {e}"
            ))
        
        return findings

    def run_infrastructure_scan(self) -> List[SecurityFinding]:
        """Run infrastructure security scan."""
        findings = []
        
        if not self.config["scanners"]["infrastructure"]["enabled"]:
            return findings

        logger.info("Running infrastructure security scan...")
        
        # Check for common infrastructure security issues
        infrastructure_files = [
            "docker-compose.yml",
            "kubernetes.yaml",
            "terraform/**/*.tf"
        ]
        
        for pattern in infrastructure_files:
            files = list(Path(".").glob(pattern))
            for file_path in files:
                findings.extend(self._scan_infrastructure_file(file_path))
        
        return findings

    def _scan_infrastructure_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan individual infrastructure file."""
        findings = []
        
        try:
            content = file_path.read_text()
            
            # Check for common security issues
            security_checks = [
                ("password", "hardcoded password", "critical"),
                ("secret", "hardcoded secret", "critical"),
                ("api_key", "hardcoded API key", "critical"),
                ("token", "hardcoded token", "critical"),
                ("privileged: true", "privileged container", "high"),
                ("--privileged", "privileged container", "high"),
                ("allowPrivilegeEscalation: true", "privilege escalation allowed", "high"),
                ("runAsRoot: true", "running as root", "medium"),
                ("hostNetwork: true", "host network access", "medium"),
                ("hostPID: true", "host PID access", "medium"),
            ]
            
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                line_lower = line.lower()
                for pattern, description, severity in security_checks:
                    if pattern in line_lower:
                        finding = SecurityFinding(
                            category="infrastructure_security",
                            severity=severity,
                            title=f"Infrastructure security issue: {description}",
                            description=f"Found {description} in {file_path}",
                            file_path=str(file_path),
                            line_number=line_num,
                            remediation="Use environment variables or secret management"
                        )
                        findings.append(finding)
                        
        except Exception as e:
            logger.error(f"Failed to scan {file_path}: {e}")
        
        return findings

    def run_privacy_specific_scan(self) -> List[SecurityFinding]:
        """Run privacy-specific security checks."""
        findings = []
        
        logger.info("Running privacy-specific security scan...")
        
        # Check for privacy-related security issues
        privacy_checks = [
            # Data handling
            ("print(.*personal_data", "Personal data in logs", "high"),
            ("logging.*personal", "Personal data in logs", "high"),
            ("console.log.*personal", "Personal data in logs", "high"),
            
            # Privacy parameters
            ("epsilon.*=.*[1-9][0-9]+", "Large epsilon value", "medium"),
            ("delta.*=.*[0-9]*\.?[0-9]*e-[0-3]", "Large delta value", "medium"),
            
            # Unsafe operations
            ("pickle.load", "Unsafe deserialization", "high"),
            ("eval\\(", "Code injection risk", "critical"),
            ("exec\\(", "Code injection risk", "critical"),
            
            # Network security
            ("verify=False", "SSL verification disabled", "high"),
            ("ssl_verify=False", "SSL verification disabled", "high"),
            ("check_cert=False", "Certificate check disabled", "high"),
        ]
        
        python_files = list(Path(".").glob("**/*.py"))
        for file_path in python_files:
            if any(exclude in str(file_path) for exclude in self.config["exclusions"]["files"]):
                continue
                
            try:
                content = file_path.read_text()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, description, severity in privacy_checks:
                        import re
                        if re.search(pattern, line):
                            finding = SecurityFinding(
                                category="privacy_security",
                                severity=severity,
                                title=f"Privacy security issue: {description}",
                                description=f"Found {description} in {file_path}:{line_num}",
                                file_path=str(file_path),
                                line_number=line_num,
                                remediation="Review and secure privacy-sensitive operations"
                            )
                            findings.append(finding)
                            
            except Exception as e:
                logger.error(f"Failed to scan {file_path}: {e}")
        
        return findings

    def run_comprehensive_scan(self) -> SecurityReport:
        """Run comprehensive security scan."""
        logger.info("Starting comprehensive security scan...")
        
        all_findings = []
        
        # Run all scanners in parallel
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(self.run_bandit_scan): "bandit",
                executor.submit(self.run_safety_scan): "safety",
                executor.submit(self.run_secrets_scan): "secrets",
                executor.submit(self.run_container_scan): "container",
                executor.submit(self.run_infrastructure_scan): "infrastructure",
                executor.submit(self.run_privacy_specific_scan): "privacy"
            }
            
            for future in as_completed(futures):
                scanner_name = futures[future]
                try:
                    findings = future.result()
                    all_findings.extend(findings)
                    logger.info(f"{scanner_name} scan completed with {len(findings)} findings")
                except Exception as e:
                    logger.error(f"{scanner_name} scan failed: {e}")
        
        # Generate summary
        summary = {
            "critical": sum(1 for f in all_findings if f.severity == "critical"),
            "high": sum(1 for f in all_findings if f.severity == "high"),
            "medium": sum(1 for f in all_findings if f.severity == "medium"),
            "low": sum(1 for f in all_findings if f.severity == "low"),
            "total": len(all_findings)
        }
        
        # Determine compliance status
        thresholds = self.config["thresholds"]
        compliance_status = "PASS"
        
        if summary["critical"] > thresholds["critical"]:
            compliance_status = "FAIL"
        elif summary["high"] > thresholds["high"]:
            compliance_status = "FAIL"
        elif summary["medium"] > thresholds["medium"]:
            compliance_status = "WARN"
        
        from datetime import datetime
        report = SecurityReport(
            timestamp=datetime.now().isoformat(),
            scan_type="comprehensive",
            findings=all_findings,
            summary=summary,
            compliance_status=compliance_status
        )
        
        logger.info(f"Security scan completed: {compliance_status}")
        logger.info(f"Findings: {summary}")
        
        return report

    def generate_reports(self, report: SecurityReport, output_dir: Path) -> None:
        """Generate security reports in multiple formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON report
        json_report = {
            "timestamp": report.timestamp,
            "scan_type": report.scan_type,
            "summary": report.summary,
            "compliance_status": report.compliance_status,
            "findings": [
                {
                    "category": f.category,
                    "severity": f.severity,
                    "title": f.title,
                    "description": f.description,
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                    "cve_id": f.cve_id,
                    "remediation": f.remediation
                }
                for f in report.findings
            ]
        }
        
        with open(output_dir / "security-report.json", 'w') as f:
            json.dump(json_report, f, indent=2)
        
        # SARIF report for GitHub integration
        sarif_report = self._generate_sarif_report(report)
        with open(output_dir / "security-report.sarif", 'w') as f:
            json.dump(sarif_report, f, indent=2)
        
        # HTML report
        html_report = self._generate_html_report(report)
        with open(output_dir / "security-report.html", 'w') as f:
            f.write(html_report)
        
        logger.info(f"Reports generated in {output_dir}")

    def _generate_sarif_report(self, report: SecurityReport) -> Dict[str, Any]:
        """Generate SARIF format report."""
        return {
            "version": "2.1.0",
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "Advanced Security Scanner",
                            "version": "1.0.0",
                            "informationUri": "https://github.com/terragon-labs/privacy-preserving-agent-finetuner"
                        }
                    },
                    "results": [
                        {
                            "ruleId": f.category,
                            "level": self._severity_to_sarif_level(f.severity),
                            "message": {"text": f.description},
                            "locations": [
                                {
                                    "physicalLocation": {
                                        "artifactLocation": {"uri": f.file_path or ""},
                                        "region": {"startLine": f.line_number or 1}
                                    }
                                }
                            ] if f.file_path else []
                        }
                        for f in report.findings
                    ]
                }
            ]
        }

    def _severity_to_sarif_level(self, severity: str) -> str:
        """Convert severity to SARIF level."""
        mapping = {
            "critical": "error",
            "high": "error",
            "medium": "warning",
            "low": "note"
        }
        return mapping.get(severity, "warning")

    def _generate_html_report(self, report: SecurityReport) -> str:
        """Generate HTML report."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Security Scan Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .finding {{ margin: 10px 0; padding: 10px; border-radius: 3px; }}
                .critical {{ background: #ffebee; border-left: 4px solid #f44336; }}
                .high {{ background: #fff3e0; border-left: 4px solid #ff9800; }}
                .medium {{ background: #f3e5f5; border-left: 4px solid #9c27b0; }}
                .low {{ background: #e8f5e8; border-left: 4px solid #4caf50; }}
            </style>
        </head>
        <body>
            <h1>Security Scan Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Scan Date: {report.timestamp}</p>
                <p>Status: {report.compliance_status}</p>
                <p>Total Findings: {report.summary['total']}</p>
                <ul>
                    <li>Critical: {report.summary['critical']}</li>
                    <li>High: {report.summary['high']}</li>
                    <li>Medium: {report.summary['medium']}</li>
                    <li>Low: {report.summary['low']}</li>
                </ul>
            </div>
            
            <h2>Findings</h2>
            {"".join([
                f'<div class="finding {f.severity}">'
                f'<h3>{f.title}</h3>'
                f'<p><strong>Severity:</strong> {f.severity.upper()}</p>'
                f'<p><strong>Category:</strong> {f.category}</p>'
                f'<p><strong>Description:</strong> {f.description}</p>'
                f'<p><strong>File:</strong> {f.file_path}:{f.line_number}</p>' if f.file_path else ''
                f'<p><strong>Remediation:</strong> {f.remediation}</p>' if f.remediation else ''
                f'</div>'
                for f in report.findings
            ])}
        </body>
        </html>
        """


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Advanced security scanner")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--output", type=Path, default=Path("security-reports"), help="Output directory")
    parser.add_argument("--format", choices=["json", "sarif", "html", "all"], default="all", help="Report format")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        scanner = AdvancedSecurityScanner(args.config)
        report = scanner.run_comprehensive_scan()
        scanner.generate_reports(report, args.output)
        
        # Print summary
        print(f"\nSecurity Scan Summary:")
        print(f"Status: {report.compliance_status}")
        print(f"Total Findings: {report.summary['total']}")
        print(f"Critical: {report.summary['critical']}")
        print(f"High: {report.summary['high']}")
        print(f"Medium: {report.summary['medium']}")
        print(f"Low: {report.summary['low']}")
        
        # Exit with error code based on compliance status
        if report.compliance_status == "FAIL":
            sys.exit(1)
        elif report.compliance_status == "WARN":
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Security scan failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()