# Docker Bake configuration for multi-architecture builds
# Usage: docker buildx bake --print

variable "REGISTRY" {
  default = "ghcr.io/terragon-labs"
}

variable "IMAGE_NAME" {
  default = "privacy-finetuner"
}

variable "VERSION" {
  default = "latest"
}

variable "PLATFORMS" {
  default = ["linux/amd64", "linux/arm64"]
}

group "default" {
  targets = ["production", "development"]
}

group "all" {
  targets = ["production", "development", "jupyter"]
}

target "base" {
  dockerfile = "Dockerfile"
  platforms = PLATFORMS
  args = {
    BUILDKIT_INLINE_CACHE = 1
  }
  cache-from = [
    "type=gha"
  ]
  cache-to = [
    "type=gha,mode=max"
  ]
}

target "production" {
  inherits = ["base"]
  target = "production"
  tags = [
    "${REGISTRY}/${IMAGE_NAME}:${VERSION}",
    "${REGISTRY}/${IMAGE_NAME}:latest"
  ]
  annotations = {
    "org.opencontainers.image.title" = "Privacy-Preserving Agent Fine-Tuner"
    "org.opencontainers.image.description" = "Enterprise-grade framework for fine-tuning LLMs with differential privacy"
    "org.opencontainers.image.vendor" = "Terragon Labs"
    "org.opencontainers.image.version" = VERSION
    "org.opencontainers.image.licenses" = "Apache-2.0"
    "org.opencontainers.image.source" = "https://github.com/terragon-labs/privacy-preserving-agent-finetuner"
    "org.opencontainers.image.documentation" = "https://docs.terragon-labs.com/privacy-finetuner"
  }
}

target "development" {
  dockerfile = "Dockerfile.dev"
  platforms = PLATFORMS
  target = "development"
  tags = [
    "${REGISTRY}/${IMAGE_NAME}:dev",
    "${REGISTRY}/${IMAGE_NAME}:development"
  ]
  args = {
    PYTHON_VERSION = "3.11"
    CUDA_VERSION = "11.8"
    BUILDKIT_INLINE_CACHE = 1
  }
  cache-from = [
    "type=gha"
  ]
  cache-to = [
    "type=gha,mode=max"
  ]
  annotations = {
    "org.opencontainers.image.title" = "Privacy-Preserving Agent Fine-Tuner (Development)"
    "org.opencontainers.image.description" = "Development environment with comprehensive tooling"
    "org.opencontainers.image.vendor" = "Terragon Labs"
    "org.opencontainers.image.version" = VERSION
    "org.opencontainers.image.licenses" = "Apache-2.0"
  }
}

target "jupyter" {
  dockerfile = "Dockerfile.jupyter"
  platforms = PLATFORMS
  tags = [
    "${REGISTRY}/${IMAGE_NAME}:jupyter",
    "${REGISTRY}/${IMAGE_NAME}:research"
  ]
  args = {
    BUILDKIT_INLINE_CACHE = 1
  }
  cache-from = [
    "type=gha"
  ]
  cache-to = [
    "type=gha,mode=max"
  ]
  annotations = {
    "org.opencontainers.image.title" = "Privacy-Preserving Agent Fine-Tuner (Jupyter)"
    "org.opencontainers.image.description" = "Jupyter environment for privacy-preserving ML research"
    "org.opencontainers.image.vendor" = "Terragon Labs"
    "org.opencontainers.image.version" = VERSION
    "org.opencontainers.image.licenses" = "Apache-2.0"
  }
}

# Security scanning target
target "security-scan" {
  inherits = ["base"]
  target = "security-scan"
  output = ["type=local,dest=./security-reports"]
  platforms = ["linux/amd64"]  # Security scan only on amd64 for performance
}

# SBOM generation target
target "sbom" {
  inherits = ["production"]
  output = ["type=local,dest=./sbom"]
  attestations = [
    "type=sbom",
    "type=provenance,mode=max"
  ]
}
