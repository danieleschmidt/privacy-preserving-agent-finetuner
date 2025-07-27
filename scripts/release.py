#!/usr/bin/env python3
"""Release automation script for Privacy-Preserving Agent Finetuner."""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReleaseManager:
    """Manages the release process for the project."""

    def __init__(self, dry_run: bool = False):
        """Initialize the release manager."""
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent.parent
        self.version_file = self.project_root / "pyproject.toml"
        
    def run_command(self, command: str, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command."""
        logger.info(f"Running: {command}")
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would run: {command}")
            return subprocess.CompletedProcess(
                command, 0, stdout=b"", stderr=b""
            )
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {command}")
            logger.error(f"Exit code: {e.returncode}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            raise

    def get_current_version(self) -> str:
        """Get the current version from pyproject.toml."""
        with open(self.version_file, 'r') as f:
            content = f.read()
        
        match = re.search(r'version = "([^"]+)"', content)
        if not match:
            raise ValueError("Could not find version in pyproject.toml")
        
        return match.group(1)

    def check_git_status(self) -> bool:
        """Check if git working directory is clean."""
        result = self.run_command("git status --porcelain")
        
        if result.stdout.strip():
            logger.error("Git working directory is not clean:")
            logger.error(result.stdout)
            return False
        
        return True

    def check_on_main_branch(self) -> bool:
        """Check if currently on main branch."""
        result = self.run_command("git branch --show-current")
        current_branch = result.stdout.strip()
        
        if current_branch != "main":
            logger.error(f"Not on main branch. Current branch: {current_branch}")
            return False
        
        return True

    def pull_latest_changes(self) -> None:
        """Pull latest changes from origin."""
        self.run_command("git fetch origin")
        self.run_command("git pull origin main")

    def run_tests(self) -> bool:
        """Run the test suite."""
        logger.info("Running test suite...")
        
        try:
            self.run_command("make test", capture_output=False)
            return True
        except subprocess.CalledProcessError:
            logger.error("Tests failed")
            return False

    def run_security_checks(self) -> bool:
        """Run security and privacy compliance checks."""
        logger.info("Running security and compliance checks...")
        
        try:
            self.run_command("make security", capture_output=False)
            self.run_command("make privacy-check", capture_output=False)
            return True
        except subprocess.CalledProcessError:
            logger.error("Security or compliance checks failed")
            return False

    def build_package(self) -> bool:
        """Build the package."""
        logger.info("Building package...")
        
        try:
            self.run_command("make build", capture_output=False)
            return True
        except subprocess.CalledProcessError:
            logger.error("Package build failed")
            return False

    def bump_version(self, bump_type: str) -> str:
        """Bump version using commitizen."""
        logger.info(f"Bumping version ({bump_type})...")
        
        current_version = self.get_current_version()
        logger.info(f"Current version: {current_version}")
        
        if bump_type == "auto":
            # Let commitizen determine the bump type
            self.run_command("poetry run cz bump --changelog")
        else:
            # Use specified bump type
            self.run_command(f"poetry run cz bump --increment {bump_type} --changelog")
        
        new_version = self.get_current_version()
        logger.info(f"New version: {new_version}")
        
        return new_version

    def create_git_tag(self, version: str) -> None:
        """Create and push git tag."""
        tag_name = f"v{version}"
        logger.info(f"Creating tag: {tag_name}")
        
        self.run_command(f"git tag -a {tag_name} -m 'Release {tag_name}'")
        self.run_command(f"git push origin {tag_name}")

    def push_changes(self) -> None:
        """Push changes to origin."""
        logger.info("Pushing changes to origin...")
        self.run_command("git push origin main")

    def publish_to_pypi(self) -> bool:
        """Publish package to PyPI."""
        logger.info("Publishing to PyPI...")
        
        try:
            self.run_command("poetry publish", capture_output=False)
            return True
        except subprocess.CalledProcessError:
            logger.error("PyPI publication failed")
            return False

    def build_and_push_docker_image(self, version: str) -> bool:
        """Build and push Docker image."""
        logger.info("Building and pushing Docker image...")
        
        try:
            registry = "ghcr.io/terragon-labs"
            image_name = "privacy-finetuner"
            
            # Build image
            self.run_command(f"docker build -t {registry}/{image_name}:{version} .")
            self.run_command(f"docker build -t {registry}/{image_name}:latest .")
            
            # Push image
            self.run_command(f"docker push {registry}/{image_name}:{version}")
            self.run_command(f"docker push {registry}/{image_name}:latest")
            
            return True
        except subprocess.CalledProcessError:
            logger.error("Docker build/push failed")
            return False

    def create_github_release(self, version: str) -> bool:
        """Create GitHub release."""
        logger.info("Creating GitHub release...")
        
        try:
            tag_name = f"v{version}"
            
            # Get changelog for this version
            changelog_section = self.get_changelog_section(version)
            
            # Create release using GitHub CLI
            self.run_command(
                f'gh release create {tag_name} '
                f'--title "Release {tag_name}" '
                f'--notes "{changelog_section}" '
                f'--generate-notes'
            )
            
            return True
        except subprocess.CalledProcessError:
            logger.error("GitHub release creation failed")
            return False

    def get_changelog_section(self, version: str) -> str:
        """Get changelog section for the given version."""
        changelog_file = self.project_root / "CHANGELOG.md"
        
        if not changelog_file.exists():
            return f"Release {version}"
        
        with open(changelog_file, 'r') as f:
            content = f.read()
        
        # Extract the section for this version
        pattern = rf"## \[{re.escape(version)}\].*?(?=## \[|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            return match.group(0).strip()
        else:
            return f"Release {version}"

    def validate_environment(self) -> bool:
        """Validate the environment before release."""
        logger.info("Validating environment...")
        
        # Check required tools
        required_tools = ["git", "poetry", "docker", "gh"]
        for tool in required_tools:
            try:
                self.run_command(f"which {tool}")
            except subprocess.CalledProcessError:
                logger.error(f"Required tool not found: {tool}")
                return False
        
        # Check git status
        if not self.check_git_status():
            return False
        
        # Check branch
        if not self.check_on_main_branch():
            return False
        
        return True

    def run_release_process(
        self,
        bump_type: str,
        skip_tests: bool = False,
        skip_security: bool = False,
        skip_pypi: bool = False,
        skip_docker: bool = False,
        skip_github: bool = False
    ) -> bool:
        """Run the complete release process."""
        logger.info("Starting release process...")
        
        try:
            # Validate environment
            if not self.validate_environment():
                return False
            
            # Pull latest changes
            self.pull_latest_changes()
            
            # Run tests
            if not skip_tests and not self.run_tests():
                return False
            
            # Run security checks
            if not skip_security and not self.run_security_checks():
                return False
            
            # Build package
            if not self.build_package():
                return False
            
            # Bump version and update changelog
            new_version = self.bump_version(bump_type)
            
            # Push changes and create tag
            self.push_changes()
            self.create_git_tag(new_version)
            
            # Publish to PyPI
            if not skip_pypi and not self.publish_to_pypi():
                logger.warning("PyPI publication failed, but continuing...")
            
            # Build and push Docker image
            if not skip_docker and not self.build_and_push_docker_image(new_version):
                logger.warning("Docker build/push failed, but continuing...")
            
            # Create GitHub release
            if not skip_github and not self.create_github_release(new_version):
                logger.warning("GitHub release creation failed, but continuing...")
            
            logger.info(f"Release {new_version} completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Release process failed: {e}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Release automation script")
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch", "auto"],
        help="Type of version bump"
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test execution")
    parser.add_argument("--skip-security", action="store_true", help="Skip security checks")
    parser.add_argument("--skip-pypi", action="store_true", help="Skip PyPI publication")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker build/push")
    parser.add_argument("--skip-github", action="store_true", help="Skip GitHub release")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.dry_run:
        logger.info("Running in DRY RUN mode - no changes will be made")
    
    release_manager = ReleaseManager(dry_run=args.dry_run)
    
    success = release_manager.run_release_process(
        bump_type=args.bump_type,
        skip_tests=args.skip_tests,
        skip_security=args.skip_security,
        skip_pypi=args.skip_pypi,
        skip_docker=args.skip_docker,
        skip_github=args.skip_github
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()