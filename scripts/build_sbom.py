#!/usr/bin/env python3
"""Generate Software Bill of Materials (SBOM) for the privacy-preserving agent finetuner."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import toml


def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], text=True
        ).strip()
        
        return {
            "commit": commit_hash,
            "branch": branch,
            "repository": remote_url
        }
    except subprocess.CalledProcessError:
        return {
            "commit": "unknown",
            "branch": "unknown", 
            "repository": "unknown"
        }


def get_poetry_dependencies() -> Dict[str, List[Dict[str, Any]]]:
    """Extract dependencies from poetry.lock file."""
    lock_file = Path("poetry.lock")
    if not lock_file.exists():
        print("Warning: poetry.lock not found. Run 'poetry lock' first.")
        return {"main": [], "dev": []}
    
    try:
        result = subprocess.run(
            ["poetry", "show", "--no-dev"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        main_deps = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    main_deps.append({
                        "name": parts[0],
                        "version": parts[1],
                        "type": "python-package"
                    })
        
        result = subprocess.run(
            ["poetry", "show", "--only", "dev"], 
            capture_output=True, 
            text=True,
            check=True
        )
        dev_deps = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    dev_deps.append({
                        "name": parts[0],
                        "version": parts[1],
                        "type": "python-package"
                    })
        
        return {"main": main_deps, "dev": dev_deps}
    
    except subprocess.CalledProcessError as e:
        print(f"Error running poetry show: {e}")
        return {"main": [], "dev": []}


def get_system_dependencies() -> List[Dict[str, Any]]:
    """Get system-level dependencies from Dockerfile."""
    dockerfile_path = Path("Dockerfile")
    system_deps = []
    
    if dockerfile_path.exists():
        with open(dockerfile_path, 'r') as f:
            content = f.read()
            
        # Parse common system packages
        common_packages = [
            "curl", "git", "build-essential", "python3", "pip"
        ]
        
        for package in common_packages:
            if package in content:
                system_deps.append({
                    "name": package,
                    "version": "latest",
                    "type": "system-package"
                })
    
    return system_deps


def get_project_metadata() -> Dict[str, Any]:
    """Extract project metadata from pyproject.toml."""
    try:
        with open("pyproject.toml", 'r') as f:
            data = toml.load(f)
        
        poetry_config = data.get("tool", {}).get("poetry", {})
        return {
            "name": poetry_config.get("name", "unknown"),
            "version": poetry_config.get("version", "0.0.0"),
            "description": poetry_config.get("description", ""),
            "authors": poetry_config.get("authors", []),
            "license": poetry_config.get("license", "unknown"),
            "homepage": poetry_config.get("homepage", ""),
            "repository": poetry_config.get("repository", ""),
        }
    except Exception as e:
        print(f"Error reading pyproject.toml: {e}")
        return {
            "name": "privacy-preserving-agent-finetuner",
            "version": "0.0.0",
            "description": "Privacy-preserving agent finetuner",
            "authors": [],
            "license": "MIT"
        }


def generate_spdx_sbom() -> Dict[str, Any]:
    """Generate SPDX-compliant SBOM."""
    git_info = get_git_info()
    dependencies = get_poetry_dependencies()
    system_deps = get_system_dependencies()
    project_meta = get_project_metadata()
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "documentName": f"{project_meta['name']}-{project_meta['version']}-sbom",
        "documentNamespace": f"https://github.com/terragon-labs/privacy-preserving-agent-finetuner/sbom/{git_info['commit']}",
        "creationInfo": {
            "created": timestamp,
            "creators": [
                "Tool: privacy-finetuner-sbom-generator",
                "Organization: Terragon Labs"
            ],
            "licenseListVersion": "3.21"
        },
        "packages": []
    }
    
    # Add main package
    sbom["packages"].append({
        "SPDXID": "SPDXRef-Package-Root",
        "name": project_meta["name"],
        "downloadLocation": project_meta.get("repository", "NOASSERTION"),
        "filesAnalyzed": False,
        "licenseConcluded": project_meta.get("license", "NOASSERTION"),
        "licenseDeclared": project_meta.get("license", "NOASSERTION"),
        "copyrightText": f"Copyright (c) 2025 {', '.join(project_meta.get('authors', ['Terragon Labs']))}",
        "versionInfo": project_meta["version"],
        "supplier": "Organization: Terragon Labs",
        "originator": "Organization: Terragon Labs"
    })
    
    # Add production dependencies
    for i, dep in enumerate(dependencies["main"]):
        sbom["packages"].append({
            "SPDXID": f"SPDXRef-Package-{dep['name']}-{i}",
            "name": dep["name"],
            "downloadLocation": f"https://pypi.org/project/{dep['name']}/{dep['version']}/",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "versionInfo": dep["version"],
            "supplier": "Organization: Python Package Index",
            "packagePurpose": "LIBRARY"
        })
    
    # Add system dependencies
    for i, dep in enumerate(system_deps):
        sbom["packages"].append({
            "SPDXID": f"SPDXRef-System-{dep['name']}-{i}",
            "name": dep["name"],
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "versionInfo": dep["version"],
            "supplier": "Organization: System Package Repository",
            "packagePurpose": "LIBRARY"
        })
    
    # Add relationships
    sbom["relationships"] = []
    
    # Root package depends on all dependencies
    for i, dep in enumerate(dependencies["main"]):
        sbom["relationships"].append({
            "spdxElementId": "SPDXRef-Package-Root",
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": f"SPDXRef-Package-{dep['name']}-{i}"
        })
    
    for i, dep in enumerate(system_deps):
        sbom["relationships"].append({
            "spdxElementId": "SPDXRef-Package-Root",
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": f"SPDXRef-System-{dep['name']}-{i}"
        })
    
    return sbom


def generate_cyclonedx_sbom() -> Dict[str, Any]:
    """Generate CycloneDX-compliant SBOM."""
    git_info = get_git_info()
    dependencies = get_poetry_dependencies()
    system_deps = get_system_dependencies()
    project_meta = get_project_metadata()
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{project_meta['name']}-{git_info['commit'][:8]}",
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": [
                {
                    "vendor": "Terragon Labs",
                    "name": "privacy-finetuner-sbom-generator",
                    "version": "1.0.0"
                }
            ],
            "component": {
                "type": "application",
                "bom-ref": f"{project_meta['name']}@{project_meta['version']}",
                "name": project_meta["name"],
                "version": project_meta["version"],
                "description": project_meta["description"],
                "licenses": [
                    {
                        "license": {
                            "id": project_meta.get("license", "MIT")
                        }
                    }
                ],
                "purl": f"pkg:pypi/{project_meta['name']}@{project_meta['version']}"
            },
            "properties": [
                {
                    "name": "git:commit",
                    "value": git_info["commit"]
                },
                {
                    "name": "git:branch", 
                    "value": git_info["branch"]
                }
            ]
        },
        "components": []
    }
    
    # Add production dependencies
    for dep in dependencies["main"]:
        sbom["components"].append({
            "type": "library",
            "bom-ref": f"{dep['name']}@{dep['version']}",
            "name": dep["name"],
            "version": dep["version"],
            "purl": f"pkg:pypi/{dep['name']}@{dep['version']}",
            "scope": "required"
        })
    
    # Add system dependencies  
    for dep in system_deps:
        sbom["components"].append({
            "type": "library",
            "bom-ref": f"{dep['name']}@{dep['version']}",
            "name": dep["name"],
            "version": dep["version"],
            "scope": "required"
        })
    
    return sbom


def main():
    """Main function to generate SBOM files."""
    output_dir = Path("dist")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating Software Bill of Materials (SBOM)...")
    
    # Generate SPDX SBOM
    print("  → Generating SPDX SBOM...")
    spdx_sbom = generate_spdx_sbom()
    spdx_path = output_dir / "sbom.spdx.json"
    with open(spdx_path, 'w') as f:
        json.dump(spdx_sbom, f, indent=2)
    print(f"    ✓ SPDX SBOM written to {spdx_path}")
    
    # Generate CycloneDX SBOM
    print("  → Generating CycloneDX SBOM...")
    cyclonedx_sbom = generate_cyclonedx_sbom()
    cyclonedx_path = output_dir / "sbom.cyclonedx.json"
    with open(cyclonedx_path, 'w') as f:
        json.dump(cyclonedx_sbom, f, indent=2)
    print(f"    ✓ CycloneDX SBOM written to {cyclonedx_path}")
    
    print("✅ SBOM generation completed successfully!")
    
    # Print summary
    spdx_packages = len(spdx_sbom["packages"])
    cyclonedx_components = len(cyclonedx_sbom["components"])
    
    print(f"\n📊 SBOM Summary:")
    print(f"  • SPDX packages: {spdx_packages}")
    print(f"  • CycloneDX components: {cyclonedx_components}")
    print(f"  • Main dependencies: {len(get_poetry_dependencies()['main'])}")
    print(f"  • System dependencies: {len(get_system_dependencies())}")


if __name__ == "__main__":
    main()