#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) Generator

Generates comprehensive SBOM in SPDX and CycloneDX formats for supply chain security.
Includes dependencies, licenses, and security vulnerability information.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import toml


class SBOMGenerator:
    """Generate Software Bill of Materials for supply chain security."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.pyproject_path = self.project_root / "pyproject.toml"
        self.output_dir = self.project_root / "sbom"
        self.output_dir.mkdir(exist_ok=True)

    def load_project_config(self) -> Dict:
        """Load project configuration from pyproject.toml."""
        if not self.pyproject_path.exists():
            raise FileNotFoundError(f"pyproject.toml not found at {self.pyproject_path}")
        
        with open(self.pyproject_path, "r") as f:
            return toml.load(f)

    def get_installed_packages(self) -> List[Dict]:
        """Get list of installed packages with versions."""
        try:
            result = subprocess.run(
                ["poetry", "show", "--tree", "--no-ansi"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.project_root
            )
            
            packages = []
            for line in result.stdout.strip().split("\n"):
                if line and not line.startswith(" "):
                    # Parse package name and version
                    parts = line.split(" ")
                    if len(parts) >= 2:
                        name = parts[0]
                        version = parts[1].strip("()")
                        packages.append({
                            "name": name,
                            "version": version,
                            "type": "python-package"
                        })
            
            return packages
        except subprocess.CalledProcessError as e:
            print(f"Error running poetry show: {e}")
            return []

    def get_vulnerabilities(self) -> Dict[str, List[Dict]]:
        """Get vulnerability information for dependencies."""
        vulnerabilities = {}
        
        try:
            # Use safety to check for known vulnerabilities
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    package_name = vuln.get("package_name", "")
                    if package_name not in vulnerabilities:
                        vulnerabilities[package_name] = []
                    
                    vulnerabilities[package_name].append({
                        "id": vuln.get("vulnerability_id", ""),
                        "description": vuln.get("advisory", ""),
                        "severity": "unknown",  # Safety doesn't provide CVSS scores
                        "affected_versions": vuln.get("vulnerable_spec", ""),
                        "fixed_versions": vuln.get("analyzed_version", "")
                    })
        
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"Warning: Could not check vulnerabilities: {e}")
        
        return vulnerabilities

    def generate_spdx_sbom(self) -> Dict:
        """Generate SBOM in SPDX format."""
        config = self.load_project_config()
        packages = self.get_installed_packages()
        vulnerabilities = self.get_vulnerabilities()
        
        # Extract project info
        project_info = config.get("tool", {}).get("poetry", {})
        project_name = project_info.get("name", "unknown")
        project_version = project_info.get("version", "unknown")
        
        spdx_doc = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": f"{project_name}-{project_version}-SBOM",
            "documentNamespace": f"https://github.com/terragon-labs/{project_name}/sbom/{project_version}",
            "creationInfo": {
                "created": datetime.now(timezone.utc).isoformat(),
                "creators": [
                    "Tool: privacy-preserving-agent-finetuner-sbom-generator",
                    "Organization: Terragon Labs"
                ],
                "licenseListVersion": "3.20"
            },
            "packages": [],
            "relationships": []
        }
        
        # Add main package
        main_package = {
            "SPDXID": "SPDXRef-Package-Root",
            "name": project_name,
            "versionInfo": project_version,
            "downloadLocation": project_info.get("repository", "NOASSERTION"),
            "filesAnalyzed": False,
            "licenseConcluded": project_info.get("license", "NOASSERTION"),
            "licenseDeclared": project_info.get("license", "NOASSERTION"),
            "copyrightText": f"Copyright (c) {datetime.now().year} Terragon Labs",
            "supplier": "Organization: Terragon Labs"
        }
        spdx_doc["packages"].append(main_package)
        
        # Add dependencies
        for i, package in enumerate(packages):
            pkg_spdx_id = f"SPDXRef-Package-{package['name']}-{i}"
            
            dep_package = {
                "SPDXID": pkg_spdx_id,
                "name": package["name"],
                "versionInfo": package["version"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
                "supplier": "NOASSERTION"
            }
            
            # Add vulnerability information if available
            if package["name"] in vulnerabilities:
                dep_package["annotations"] = []
                for vuln in vulnerabilities[package["name"]]:
                    dep_package["annotations"].append({
                        "annotationType": "SECURITY",
                        "annotator": "Tool: safety",
                        "annotationDate": datetime.now(timezone.utc).isoformat(),
                        "annotationComment": f"Vulnerability {vuln['id']}: {vuln['description']}"
                    })
            
            spdx_doc["packages"].append(dep_package)
            
            # Add relationship
            spdx_doc["relationships"].append({
                "spdxElementId": "SPDXRef-Package-Root",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": pkg_spdx_id
            })
        
        return spdx_doc

    def generate_cyclonedx_sbom(self) -> Dict:
        """Generate SBOM in CycloneDX format."""
        config = self.load_project_config()
        packages = self.get_installed_packages()
        vulnerabilities = self.get_vulnerabilities()
        
        # Extract project info
        project_info = config.get("tool", {}).get("poetry", {})
        project_name = project_info.get("name", "unknown")
        project_version = project_info.get("version", "unknown")
        
        cyclonedx_doc = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": f"urn:uuid:{project_name}-{project_version}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "version": 1,
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tools": [
                    {
                        "vendor": "Terragon Labs",
                        "name": "privacy-preserving-agent-finetuner-sbom-generator",
                        "version": "1.0.0"
                    }
                ],
                "component": {
                    "type": "application",
                    "bom-ref": "pkg:pypi/privacy-preserving-agent-finetuner@0.1.0",
                    "name": project_name,
                    "version": project_version,
                    "description": project_info.get("description", ""),
                    "licenses": [
                        {
                            "license": {
                                "id": project_info.get("license", "MIT")
                            }
                        }
                    ]
                }
            },
            "components": [],
            "vulnerabilities": []
        }
        
        # Add dependencies as components
        for package in packages:
            component = {
                "type": "library",
                "bom-ref": f"pkg:pypi/{package['name']}@{package['version']}",
                "name": package["name"],
                "version": package["version"],
                "purl": f"pkg:pypi/{package['name']}@{package['version']}",
                "scope": "required"
            }
            cyclonedx_doc["components"].append(component)
            
            # Add vulnerabilities
            if package["name"] in vulnerabilities:
                for vuln in vulnerabilities[package["name"]]:
                    vulnerability = {
                        "id": vuln["id"],
                        "source": {
                            "name": "PyUp Safety DB",
                            "url": "https://pyup.io/safety/"
                        },
                        "description": vuln["description"],
                        "affects": [
                            {
                                "ref": f"pkg:pypi/{package['name']}@{package['version']}"
                            }
                        ]
                    }
                    cyclonedx_doc["vulnerabilities"].append(vulnerability)
        
        return cyclonedx_doc

    def generate_all_formats(self) -> None:
        """Generate SBOM in all supported formats."""
        print("Generating Software Bill of Materials (SBOM)...")
        
        # Generate SPDX format
        print("Generating SPDX SBOM...")
        spdx_sbom = self.generate_spdx_sbom()
        spdx_path = self.output_dir / "sbom.spdx.json"
        with open(spdx_path, "w") as f:
            json.dump(spdx_sbom, f, indent=2)
        print(f"SPDX SBOM generated: {spdx_path}")
        
        # Generate CycloneDX format
        print("Generating CycloneDX SBOM...")
        cyclonedx_sbom = self.generate_cyclonedx_sbom()
        cyclonedx_path = self.output_dir / "sbom.cyclonedx.json"
        with open(cyclonedx_path, "w") as f:
            json.dump(cyclonedx_sbom, f, indent=2)
        print(f"CycloneDX SBOM generated: {cyclonedx_path}")
        
        # Generate summary report
        self.generate_summary_report(spdx_sbom, cyclonedx_sbom)
        
        print("\nSBOM generation completed successfully!")
        print("These files can be used for:")
        print("- Supply chain security analysis")
        print("- License compliance verification")
        print("- Vulnerability scanning")
        print("- Regulatory compliance (NIST, EU Cyber Resilience Act)")

    def generate_summary_report(self, spdx_data: Dict, cyclonedx_data: Dict) -> None:
        """Generate human-readable summary report."""
        total_packages = len(spdx_data["packages"]) - 1  # Exclude root package
        total_vulnerabilities = len(cyclonedx_data["vulnerabilities"])
        
        summary = f"""
# SBOM Summary Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## Overview
- **Total Dependencies**: {total_packages}
- **Known Vulnerabilities**: {total_vulnerabilities}
- **SPDX Version**: {spdx_data['spdxVersion']}
- **CycloneDX Version**: {cyclonedx_data['specVersion']}

## Security Status
{"⚠️  VULNERABILITIES DETECTED" if total_vulnerabilities > 0 else "✅ NO KNOWN VULNERABILITIES"}

## Usage
1. Upload SBOM files to your security scanning platform
2. Review vulnerability reports and update affected packages
3. Include SBOM in your release artifacts
4. Use for compliance reporting (NIST SSDF, EU CRA)

## Files Generated
- `sbom.spdx.json` - SPDX 2.3 format for compliance
- `sbom.cyclonedx.json` - CycloneDX 1.4 format for security tools
- `sbom_summary.md` - This human-readable summary

## Next Steps
1. Review and address any identified vulnerabilities
2. Integrate SBOM generation into CI/CD pipeline
3. Store SBOMs in artifact repository
4. Set up automated vulnerability monitoring
"""
        
        summary_path = self.output_dir / "sbom_summary.md"
        with open(summary_path, "w") as f:
            f.write(summary.strip())
        print(f"Summary report generated: {summary_path}")


def main():
    """Main entry point for SBOM generation."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    generator = SBOMGenerator(project_root)
    
    try:
        generator.generate_all_formats()
    except Exception as e:
        print(f"Error generating SBOM: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()