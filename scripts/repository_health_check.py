#!/usr/bin/env python3
"""
Repository health monitoring and automated maintenance script.

This script performs comprehensive health checks on the repository,
including dependency updates, security scans, and maintenance tasks.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any


class RepositoryHealthChecker:
    """Comprehensive repository health monitoring."""
    
    def __init__(self, repo_path: Path = None):
        self.repo_path = repo_path or Path.cwd()
        self.health_report = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "recommendations": [],
            "critical_issues": [],
            "warnings": []
        }
    
    def check_dependency_freshness(self) -> Dict[str, Any]:
        """Check if dependencies are up to date."""
        pyproject_file = self.repo_path / "pyproject.toml"
        if not pyproject_file.exists():
            return {"status": "error", "message": "pyproject.toml not found"}
        
        try:
            # Check for outdated packages
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                return {
                    "status": "success",
                    "outdated_count": len(outdated),
                    "outdated_packages": outdated[:10],  # Limit to first 10
                    "needs_update": len(outdated) > 0
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
        return {"status": "unknown", "message": "Could not determine dependency status"}
    
    def check_security_vulnerabilities(self) -> Dict[str, Any]:
        """Check for known security vulnerabilities."""
        try:
            # Run safety check
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "vulnerabilities": [],
                    "safe": True
                }
            else:
                if result.stdout:
                    vulns = json.loads(result.stdout)
                    return {
                        "status": "warning",
                        "vulnerabilities": vulns,
                        "safe": False,
                        "count": len(vulns)
                    }
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
        return {"status": "unknown", "message": "Could not check vulnerabilities"}
    
    def check_code_quality(self) -> Dict[str, Any]:
        """Check code quality metrics."""
        quality_report = {
            "linting": self._check_linting(),
            "formatting": self._check_formatting(),
            "type_checking": self._check_type_checking(),
            "complexity": self._check_complexity()
        }
        
        issues_count = sum(
            check.get("issues", 0) for check in quality_report.values()
            if isinstance(check.get("issues"), int)
        )
        
        return {
            "status": "success" if issues_count == 0 else "warning",
            "total_issues": issues_count,
            "details": quality_report
        }
    
    def _check_linting(self) -> Dict[str, Any]:
        """Check linting with ruff."""
        try:
            result = subprocess.run(
                ["ruff", "check", "privacy_finetuner", "--output-format", "json"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.stdout:
                issues = json.loads(result.stdout)
                return {
                    "status": "warning" if issues else "success",
                    "issues": len(issues),
                    "details": issues[:5]  # First 5 issues
                }
            return {"status": "success", "issues": 0}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _check_formatting(self) -> Dict[str, Any]:
        """Check code formatting with black."""
        try:
            result = subprocess.run(
                ["black", "--check", "privacy_finetuner"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode == 0:
                return {"status": "success", "formatted": True}
            else:
                return {
                    "status": "warning",
                    "formatted": False,
                    "message": "Code needs formatting"
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _check_type_checking(self) -> Dict[str, Any]:
        """Check type annotations with mypy."""
        try:
            result = subprocess.run(
                ["mypy", "privacy_finetuner", "--json-report", "/tmp/mypy-report"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            report_file = Path("/tmp/mypy-report/index.txt")
            if report_file.exists():
                with open(report_file) as f:
                    content = f.read()
                    # Parse mypy output for errors
                    errors = content.count("error")
                    return {
                        "status": "success" if errors == 0 else "warning",
                        "errors": errors
                    }
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
        return {"status": "skipped", "message": "Type checking not configured"}
    
    def _check_complexity(self) -> Dict[str, Any]:
        """Check code complexity with radon."""
        try:
            result = subprocess.run(
                ["radon", "cc", "privacy_finetuner", "-j"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.returncode == 0 and result.stdout:
                complexity_data = json.loads(result.stdout)
                high_complexity = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item.get("type") == "function" and item.get("complexity", 0) > 10:
                            high_complexity += 1
                
                return {
                    "status": "warning" if high_complexity > 0 else "success",
                    "high_complexity_functions": high_complexity
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
        return {"status": "skipped", "message": "Complexity checking not available"}
    
    def check_test_coverage(self) -> Dict[str, Any]:
        """Check test coverage."""
        try:
            result = subprocess.run(
                ["pytest", "--cov=privacy_finetuner", "--cov-report=json", "--tb=no"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            coverage_file = self.repo_path / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                target_coverage = 85  # From metrics config
                
                return {
                    "status": "success" if total_coverage >= target_coverage else "warning",
                    "coverage_percent": total_coverage,
                    "target_percent": target_coverage,
                    "meets_target": total_coverage >= target_coverage
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
        return {"status": "unknown", "message": "Could not determine test coverage"}
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness."""
        required_docs = [
            "README.md", "CONTRIBUTING.md", "SECURITY.md",
            "docs/ARCHITECTURE.md", "docs/ROADMAP.md"
        ]
        
        missing_docs = []
        for doc in required_docs:
            if not (self.repo_path / doc).exists():
                missing_docs.append(doc)
        
        # Check if Python modules have docstrings
        py_files = list(self.repo_path.glob("privacy_finetuner/**/*.py"))
        undocumented_modules = []
        
        for py_file in py_files:
            if py_file.name != "__init__.py":
                try:
                    with open(py_file) as f:
                        content = f.read()
                        # Simple check for module docstring
                        if not ('"""' in content[:200] or "'''" in content[:200]):
                            undocumented_modules.append(str(py_file.relative_to(self.repo_path)))
                except Exception:
                    continue
        
        return {
            "status": "success" if not missing_docs and not undocumented_modules else "warning",
            "missing_documents": missing_docs,
            "undocumented_modules": undocumented_modules[:5],  # Limit to 5
            "documentation_score": max(0, 100 - len(missing_docs) * 10 - len(undocumented_modules) * 2)
        }
    
    def check_git_health(self) -> Dict[str, Any]:
        """Check Git repository health."""
        try:
            # Check for uncommitted changes
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            uncommitted_files = status_result.stdout.strip().split('\n') if status_result.stdout.strip() else []
            
            # Check branch status
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            current_branch = branch_result.stdout.strip()
            
            # Check if ahead/behind remote
            remote_check = subprocess.run(
                ["git", "status", "-b", "--porcelain"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            return {
                "status": "success",
                "current_branch": current_branch,
                "uncommitted_changes": len(uncommitted_files) > 0,
                "uncommitted_count": len(uncommitted_files),
                "clean_working_directory": len(uncommitted_files) == 0
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on health checks."""
        recommendations = []
        
        # Check dependency freshness
        deps = self.health_report["checks"].get("dependencies", {})
        if deps.get("needs_update"):
            recommendations.append(
                f"Update {deps.get('outdated_count', 0)} outdated dependencies"
            )
        
        # Check security
        security = self.health_report["checks"].get("security", {})
        if not security.get("safe", True):
            recommendations.append(
                f"Address {security.get('count', 0)} security vulnerabilities"
            )
        
        # Check code quality
        quality = self.health_report["checks"].get("code_quality", {})
        if quality.get("total_issues", 0) > 0:
            recommendations.append(
                f"Fix {quality.get('total_issues')} code quality issues"
            )
        
        # Check test coverage
        coverage = self.health_report["checks"].get("test_coverage", {})
        if not coverage.get("meets_target", True):
            recommendations.append(
                f"Increase test coverage from {coverage.get('coverage_percent', 0)}% to {coverage.get('target_percent', 85)}%"
            )
        
        # Check documentation
        docs = self.health_report["checks"].get("documentation", {})
        if docs.get("missing_documents"):
            recommendations.append(
                f"Add missing documentation: {', '.join(docs.get('missing_documents', [])[:3])}"
            )
        
        return recommendations
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        print("Running repository health check...")
        
        # Run all checks
        self.health_report["checks"] = {
            "dependencies": self.check_dependency_freshness(),
            "security": self.check_security_vulnerabilities(),
            "code_quality": self.check_code_quality(),
            "test_coverage": self.check_test_coverage(),
            "documentation": self.check_documentation(),
            "git_health": self.check_git_health()
        }
        
        # Generate recommendations
        self.health_report["recommendations"] = self.generate_recommendations()
        
        # Determine overall health status
        critical_issues = []
        warnings = []
        
        for check_name, check_result in self.health_report["checks"].items():
            if check_result.get("status") == "error":
                critical_issues.append(f"{check_name}: {check_result.get('message', 'Unknown error')}")
            elif check_result.get("status") == "warning":
                warnings.append(f"{check_name}: Issues detected")
        
        self.health_report["critical_issues"] = critical_issues
        self.health_report["warnings"] = warnings
        
        # Calculate overall health score
        total_checks = len(self.health_report["checks"])
        passed_checks = sum(
            1 for check in self.health_report["checks"].values()
            if check.get("status") == "success"
        )
        
        self.health_report["health_score"] = int((passed_checks / total_checks) * 100) if total_checks > 0 else 0
        
        return self.health_report
    
    def save_report(self, filename: str = "health-report.json") -> Path:
        """Save health report to file."""
        report_path = self.repo_path / filename
        with open(report_path, "w") as f:
            json.dump(self.health_report, f, indent=2)
        return report_path


def main():
    """Main entry point for repository health check."""
    if len(sys.argv) > 1:
        repo_path = Path(sys.argv[1])
    else:
        repo_path = Path.cwd()
    
    checker = RepositoryHealthChecker(repo_path)
    report = checker.run_health_check()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"REPOSITORY HEALTH REPORT")
    print(f"{'='*60}")
    print(f"Overall Health Score: {report['health_score']}%")
    
    if report['critical_issues']:
        print(f"\n❌ CRITICAL ISSUES ({len(report['critical_issues'])}):")
        for issue in report['critical_issues']:
            print(f"  - {issue}")
    
    if report['warnings']:
        print(f"\n⚠️  WARNINGS ({len(report['warnings'])}):")
        for warning in report['warnings']:
            print(f"  - {warning}")
    
    if report['recommendations']:
        print(f"\n📝 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Save report
    report_file = checker.save_report()
    print(f"\nDetailed report saved to: {report_file}")
    
    # Exit with appropriate code
    if report['critical_issues']:
        sys.exit(2)  # Critical issues
    elif report['warnings']:
        sys.exit(1)  # Warnings
    else:
        print("\n✅ Repository health check passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()