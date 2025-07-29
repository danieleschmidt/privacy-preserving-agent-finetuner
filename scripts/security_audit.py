#!/usr/bin/env python3
"""
Comprehensive Security Audit Script

Performs multi-layer security analysis including:
- Static code analysis
- Dependency vulnerability scanning
- Container security scanning
- Configuration security review
- Privacy compliance verification
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SecurityAuditor:
    """Comprehensive security audit engine."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root.absolute()),
            "audits": {},
            "summary": {
                "total_issues": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "info": 0
            }
        }

    def run_bandit_scan(self) -> Dict:
        """Run Bandit static security analysis."""
        print("🔍 Running Bandit static security analysis...")
        
        try:
            result = subprocess.run([
                "bandit",
                "-r", str(self.project_root),
                "-f", "json",
                "-o", "/tmp/bandit_results.json",
                "--exclude", "tests/,venv/,.venv/,build/,dist/"
            ], capture_output=True, text=True)
            
            if Path("/tmp/bandit_results.json").exists():
                with open("/tmp/bandit_results.json", "r") as f:
                    bandit_data = json.loads(f.read())
                    
                return {
                    "tool": "bandit",
                    "status": "completed",
                    "issues_found": len(bandit_data.get("results", [])),
                    "results": bandit_data.get("results", []),
                    "metrics": bandit_data.get("metrics", {}),
                    "raw_output": result.stdout
                }
        except Exception as e:
            return {
                "tool": "bandit",
                "status": "error",
                "error": str(e),
                "issues_found": 0
            }

    def run_safety_scan(self) -> Dict:
        """Run Safety dependency vulnerability scan."""
        print("🛡️  Running Safety dependency vulnerability scan...")
        
        try:
            result = subprocess.run([
                "safety", "check", "--json", "--full-report"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                return {
                    "tool": "safety",
                    "status": "completed",
                    "issues_found": len(safety_data),
                    "vulnerabilities": safety_data,
                    "raw_output": result.stdout
                }
            else:
                return {
                    "tool": "safety",
                    "status": "no_vulnerabilities",
                    "issues_found": 0,
                    "message": "No known security vulnerabilities found"
                }
                
        except Exception as e:
            return {
                "tool": "safety",
                "status": "error",
                "error": str(e),
                "issues_found": 0
            }

    def run_docker_security_scan(self) -> Dict:
        """Run Docker container security scan using Trivy."""
        print("🐳 Running Docker security scan...")
        
        dockerfile_path = self.project_root / "Dockerfile"
        if not dockerfile_path.exists():
            return {
                "tool": "trivy",
                "status": "skipped",
                "reason": "No Dockerfile found",
                "issues_found": 0
            }
        
        try:
            # Build image first
            build_result = subprocess.run([
                "docker", "build", "-t", "privacy-finetuner-security-scan", "."
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if build_result.returncode != 0:
                return {
                    "tool": "trivy",
                    "status": "build_failed",
                    "error": build_result.stderr,
                    "issues_found": 0
                }
            
            # Run Trivy scan
            trivy_result = subprocess.run([
                "trivy", "image", "--format", "json", 
                "--severity", "CRITICAL,HIGH,MEDIUM",
                "privacy-finetuner-security-scan"
            ], capture_output=True, text=True)
            
            if trivy_result.stdout:
                trivy_data = json.loads(trivy_result.stdout)
                total_vulnerabilities = 0
                if "Results" in trivy_data:
                    for result in trivy_data["Results"]:
                        if "Vulnerabilities" in result:
                            total_vulnerabilities += len(result["Vulnerabilities"])
                
                return {
                    "tool": "trivy",
                    "status": "completed",
                    "issues_found": total_vulnerabilities,
                    "results": trivy_data,
                    "raw_output": trivy_result.stdout
                }
                
        except Exception as e:
            return {
                "tool": "trivy",
                "status": "error",
                "error": str(e),
                "issues_found": 0
            }

    def audit_secrets_detection(self) -> Dict:
        """Run secrets detection scan."""
        print("🔐 Running secrets detection scan...")
        
        try:
            result = subprocess.run([
                "detect-secrets", "scan", "--all-files", 
                "--baseline", ".secrets.baseline",
                "--force-use-all-plugins"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                secrets_data = json.loads(result.stdout)
                total_secrets = sum(len(files) for files in secrets_data.get("results", {}).values())
                
                return {
                    "tool": "detect-secrets",
                    "status": "completed",
                    "issues_found": total_secrets,
                    "results": secrets_data,
                    "raw_output": result.stdout
                }
            else:
                return {
                    "tool": "detect-secrets",
                    "status": "no_secrets",
                    "issues_found": 0,
                    "message": "No secrets detected"
                }
                
        except Exception as e:
            return {
                "tool": "detect-secrets",
                "status": "error",
                "error": str(e),
                "issues_found": 0
            }

    def audit_configuration_security(self) -> Dict:
        """Audit configuration files for security issues."""
        print("⚙️  Auditing configuration security...")
        
        config_issues = []
        security_configs = [
            ("docker-compose.yml", self.check_docker_compose_security),
            (".env.example", self.check_env_security),
            ("pyproject.toml", self.check_python_config_security),
        ]
        
        for config_file, check_function in security_configs:
            config_path = self.project_root / config_file
            if config_path.exists():
                issues = check_function(config_path)
                config_issues.extend(issues)
        
        return {
            "tool": "config-audit",
            "status": "completed",
            "issues_found": len(config_issues),
            "issues": config_issues
        }

    def check_docker_compose_security(self, file_path: Path) -> List[Dict]:
        """Check Docker Compose configuration for security issues."""
        issues = []
        try:
            content = file_path.read_text()
            
            # Check for security anti-patterns
            if "privileged: true" in content:
                issues.append({
                    "file": str(file_path),
                    "severity": "high",
                    "issue": "Privileged container detected",
                    "description": "Running containers in privileged mode increases security risk"
                })
            
            if "--privileged" in content:
                issues.append({
                    "file": str(file_path),
                    "severity": "high",
                    "issue": "Privileged flag detected",
                    "description": "Privileged containers have elevated access to host"
                })
            
            if "network_mode: host" in content:
                issues.append({
                    "file": str(file_path),
                    "severity": "medium",
                    "issue": "Host network mode",
                    "description": "Host networking reduces container isolation"
                })
                
            if ":latest" in content:
                issues.append({
                    "file": str(file_path),
                    "severity": "low",
                    "issue": "Latest tag usage",
                    "description": "Using 'latest' tag can lead to unpredictable deployments"
                })
                
        except Exception as e:
            issues.append({
                "file": str(file_path),
                "severity": "info",
                "issue": "Failed to parse Docker Compose file",
                "description": str(e)
            })
        
        return issues

    def check_env_security(self, file_path: Path) -> List[Dict]:
        """Check environment configuration for security issues."""
        issues = []
        try:
            content = file_path.read_text()
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check for hardcoded secrets
                if any(keyword in line.lower() for keyword in 
                       ['password=', 'secret=', 'key=', 'token=']):
                    if not any(placeholder in line.lower() for placeholder in 
                              ['your-', 'change-', 'example-', 'placeholder']):
                        issues.append({
                            "file": str(file_path),
                            "line": line_num,
                            "severity": "medium",
                            "issue": "Potential hardcoded secret",
                            "description": f"Line contains what appears to be a hardcoded secret: {line[:50]}..."
                        })
                
                # Check for debug mode in production
                if "DEBUG=true" in line or "DEBUG=True" in line:
                    issues.append({
                        "file": str(file_path),
                        "line": line_num,
                        "severity": "low",
                        "issue": "Debug mode enabled",
                        "description": "Debug mode should be disabled in production"
                    })
                
        except Exception as e:
            issues.append({
                "file": str(file_path),
                "severity": "info",
                "issue": "Failed to parse environment file",
                "description": str(e)
            })
        
        return issues

    def check_python_config_security(self, file_path: Path) -> List[Dict]:
        """Check Python project configuration for security issues."""
        issues = []
        try:
            import toml
            config = toml.load(file_path)
            
            # Check for insecure dependencies
            dependencies = config.get("tool", {}).get("poetry", {}).get("dependencies", {})
            
            # Example security checks
            for dep_name, dep_version in dependencies.items():
                if isinstance(dep_version, str) and "*" in dep_version:
                    issues.append({
                        "file": str(file_path),
                        "severity": "low",
                        "issue": f"Wildcard version for {dep_name}",
                        "description": "Wildcard versions can introduce unexpected changes"
                    })
            
        except Exception as e:
            issues.append({
                "file": str(file_path),
                "severity": "info",
                "issue": "Failed to parse Python configuration",
                "description": str(e)
            })
        
        return issues

    def run_privacy_compliance_audit(self) -> Dict:
        """Run privacy compliance audit."""
        print("🔒 Running privacy compliance audit...")
        
        try:
            # Run the existing privacy compliance check
            result = subprocess.run([
                "python", "scripts/privacy_compliance_check.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            return {
                "tool": "privacy-compliance",
                "status": "completed",
                "exit_code": result.returncode,
                "output": result.stdout,
                "errors": result.stderr,
                "compliant": result.returncode == 0
            }
            
        except Exception as e:
            return {
                "tool": "privacy-compliance",
                "status": "error",
                "error": str(e)
            }

    def calculate_risk_score(self) -> int:
        """Calculate overall risk score based on findings."""
        score = 0
        
        # Weight issues by severity
        severity_weights = {
            "critical": 10,
            "high": 7,
            "medium": 4,
            "low": 2,
            "info": 1
        }
        
        for audit_name, audit_result in self.results["audits"].items():
            if isinstance(audit_result, dict) and "issues_found" in audit_result:
                issues_found = audit_result["issues_found"]
                
                # Simple scoring - could be more sophisticated
                if "issues" in audit_result:
                    for issue in audit_result["issues"]:
                        severity = issue.get("severity", "low")
                        score += severity_weights.get(severity, 1)
                else:
                    # Generic scoring for tools without detailed issue breakdown
                    score += issues_found * severity_weights["medium"]
        
        return min(score, 100)  # Cap at 100

    def generate_report(self) -> None:
        """Generate comprehensive security audit report."""
        risk_score = self.calculate_risk_score()
        
        # Determine risk level
        if risk_score >= 80:
            risk_level = "CRITICAL"
            risk_color = "🔴"
        elif risk_score >= 60:
            risk_level = "HIGH"
            risk_color = "🟠"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
            risk_color = "🟡"
        elif risk_score >= 20:
            risk_level = "LOW"
            risk_color = "🟢"
        else:
            risk_level = "MINIMAL"
            risk_color = "✅"
        
        report = f"""
# Security Audit Report

**Generated**: {self.results['timestamp']}
**Project**: {self.results['project_root']}
**Risk Score**: {risk_score}/100 {risk_color}
**Risk Level**: {risk_level}

## Executive Summary

This comprehensive security audit analyzed the privacy-preserving agent finetuner 
for potential security vulnerabilities, misconfigurations, and compliance issues.

### Overall Findings
- **Total Issues**: {sum(audit.get('issues_found', 0) for audit in self.results['audits'].values() if isinstance(audit, dict))}
- **Risk Level**: {risk_level}
- **Compliance Status**: {"✅ COMPLIANT" if self.results['audits'].get('privacy-compliance', {}).get('compliant', False) else "❌ NON-COMPLIANT"}

## Audit Results

"""
        
        for audit_name, audit_result in self.results["audits"].items():
            if isinstance(audit_result, dict):
                status = audit_result.get("status", "unknown")
                issues_found = audit_result.get("issues_found", 0)
                
                status_emoji = {
                    "completed": "✅",
                    "error": "❌",
                    "skipped": "⏭️",
                    "no_vulnerabilities": "✅",
                    "no_secrets": "✅"
                }.get(status, "❓")
                
                report += f"### {audit_name.title()} {status_emoji}\n"
                report += f"- **Status**: {status}\n"
                report += f"- **Issues Found**: {issues_found}\n"
                
                if "error" in audit_result:
                    report += f"- **Error**: {audit_result['error']}\n"
                
                report += "\n"
        
        report += """
## Recommendations

1. **Critical Issues**: Address all critical and high-severity findings immediately
2. **Dependency Updates**: Keep all dependencies updated to latest secure versions
3. **Regular Scanning**: Integrate security scanning into CI/CD pipeline
4. **Privacy Compliance**: Ensure all privacy compliance checks pass
5. **Configuration Review**: Regular review of security configurations

## Next Steps

1. Review detailed findings in the JSON report
2. Create tickets for all medium+ severity issues
3. Implement automated security scanning in CI/CD
4. Schedule regular security audits
5. Update security policies and procedures

---
*This report was generated automatically. For questions, contact the security team.*
"""
        
        # Save reports
        report_dir = self.project_root / "security_audit_reports"
        report_dir.mkdir(exist_ok=True)
        
        # Save human-readable report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        readable_report_path = report_dir / f"security_audit_{timestamp}.md"
        with open(readable_report_path, "w") as f:
            f.write(report.strip())
        
        # Save detailed JSON report
        json_report_path = report_dir / f"security_audit_{timestamp}.json"
        with open(json_report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📊 Security audit completed!")
        print(f"📄 Readable report: {readable_report_path}")
        print(f"📋 Detailed report: {json_report_path}")
        print(f"🎯 Risk Score: {risk_score}/100 ({risk_level})")
        
        # Return non-zero exit code for high risk
        if risk_score >= 60:
            print(f"\n⚠️  HIGH RISK DETECTED - Please review findings immediately")
            return 1
        
        return 0

    def run_full_audit(self) -> int:
        """Run comprehensive security audit."""
        print("🛡️  Starting comprehensive security audit...\n")
        
        # Run all security audits
        self.results["audits"]["static_analysis"] = self.run_bandit_scan()
        self.results["audits"]["dependency_scan"] = self.run_safety_scan()
        self.results["audits"]["container_scan"] = self.run_docker_security_scan()
        self.results["audits"]["secrets_detection"] = self.audit_secrets_detection()
        self.results["audits"]["configuration_audit"] = self.audit_configuration_security()
        self.results["audits"]["privacy_compliance"] = self.run_privacy_compliance_audit()
        
        return self.generate_report()


def main():
    """Main entry point for security audit."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    auditor = SecurityAuditor(project_root)
    exit_code = auditor.run_full_audit()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()