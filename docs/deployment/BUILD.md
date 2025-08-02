# Build and Containerization Guide

This guide covers building, containerizing, and deploying the Privacy-Preserving Agent Fine-Tuner.

## Build System Overview

The project uses a comprehensive build system with:

- **Multi-stage Docker builds** for production optimization
- **Development containers** with full tooling
- **Jupyter environments** for research
- **Multi-architecture support** (AMD64, ARM64)
- **Security scanning** integration
- **SBOM generation** for supply chain security

## Container Images

### Production Image (`Dockerfile`)

```bash
# Build production image
docker build -t privacy-finetuner:latest .

# Build specific target
docker build --target production -t privacy-finetuner:prod .
```

Features:
- Multi-stage build for minimal size
- Non-root user execution
- Security hardening
- Health checks
- Proper label metadata

### Development Image (`Dockerfile.dev`)

```bash
# Build development image
docker build -f Dockerfile.dev -t privacy-finetuner:dev .

# With build arguments
docker build -f Dockerfile.dev \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg CUDA_VERSION=11.8 \
  -t privacy-finetuner:dev .
```

Features:
- CUDA support for GPU development
- Comprehensive development tools
- Hot reload capabilities
- Debugging tools
- Jupyter integration

### Jupyter Research Image (`Dockerfile.jupyter`)

```bash
# Build Jupyter image
docker build -f Dockerfile.jupyter -t privacy-finetuner:jupyter .
```

Features:
- Pre-configured Jupyter Lab
- Privacy ML libraries
- Example notebooks
- Research-focused environment

## Multi-Architecture Builds

Using Docker Buildx for multi-platform builds:

```bash
# Setup buildx builder
docker buildx create --name privacy-builder --use

# Build for multiple architectures
docker buildx bake --print  # Show configuration
docker buildx bake all       # Build all targets
docker buildx bake production # Build production only

# Push to registry
docker buildx bake --push production
```

### Supported Platforms

- `linux/amd64` - Primary production platform
- `linux/arm64` - ARM64 support for modern servers

## Docker Compose Environments

### Production Stack (`docker-compose.yml`)

```bash
# Start full production stack
docker-compose up -d

# Scale specific services
docker-compose up -d --scale privacy-finetuner=3

# View logs
docker-compose logs -f privacy-finetuner
```

Includes:
- Privacy Fine-tuner API
- PostgreSQL database
- Redis cache
- Prometheus monitoring
- Grafana dashboards
- Jaeger tracing
- MinIO object storage
- Nginx reverse proxy

### Development Stack (`docker-compose.dev.yml`)

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Start with GPU support
docker-compose -f docker-compose.dev.yml --profile gpu up -d

# Access development tools
docker-compose -f docker-compose.dev.yml exec privacy-finetuner-dev bash
```

Includes:
- Development container with hot reload
- GPU support configuration
- Jupyter Lab environment
- Development databases
- Monitoring stack
- Mail testing (MailHog)

## Build Commands

### Make Commands

```bash
# Build package
make build

# Build Docker images
make docker-build
make docker-build-dev

# Run containers
make docker-run
make docker-run-dev

# Push to registry
make docker-push

# Compose operations
make docker-compose-up
make docker-compose-down
```

### Poetry Commands

```bash
# Install dependencies
poetry install
poetry install --with dev,docs

# Build package
poetry build

# Publish to PyPI
poetry publish
```

## Security and Compliance

### Container Security Scanning

```bash
# Scan with Trivy
trivy image privacy-finetuner:latest

# Scan with custom config
trivy image --config trivy.yaml privacy-finetuner:latest

# Generate SARIF report
trivy image --format sarif -o results.sarif privacy-finetuner:latest
```

### SBOM Generation

```bash
# Generate Software Bill of Materials
python scripts/generate_sbom.py

# Docker SBOM with buildkit
docker buildx bake sbom

# View SBOM
cat sbom.json | jq '.packages[] | {name, version}'
```

### Security Best Practices

1. **Non-root execution**: All containers run as non-root users
2. **Minimal base images**: Using slim/alpine variants
3. **Multi-stage builds**: Excluding development dependencies
4. **Security scanning**: Automated vulnerability detection
5. **Secrets management**: Using environment variables and secrets
6. **Network isolation**: Proper network segmentation
7. **Resource limits**: CPU and memory constraints

## Performance Optimization

### Build Optimization

1. **Layer caching**: Optimized Dockerfile layer order
2. **Multi-stage builds**: Separate build and runtime stages
3. **Dependency caching**: Poetry cache optimization
4. **Parallel builds**: Multi-architecture parallel builds

### Runtime Optimization

1. **Resource allocation**: Proper CPU/memory limits
2. **GPU scheduling**: NVIDIA GPU resource management
3. **Volume mounting**: Efficient data access patterns
4. **Network optimization**: Service mesh considerations

## Deployment Strategies

### Local Development

```bash
# Quick start
make setup-dev
docker-compose -f docker-compose.dev.yml up -d

# Access services
open http://localhost:8080     # API
open http://localhost:8888     # Jupyter
open http://localhost:3001     # Grafana
```

### Staging Environment

```bash
# Deploy to staging
make deploy-staging

# Health check
curl -f http://staging.privacy-finetuner.com/health
```

### Production Deployment

```bash
# Deploy to production
make deploy-prod

# Rolling update
docker service update --image privacy-finetuner:v1.2.0 privacy_finetuner

# Rollback
make rollback
```

## Monitoring and Observability

### Container Metrics

- **Resource usage**: CPU, memory, disk, network
- **Application metrics**: Request rates, errors, latency
- **Privacy metrics**: Privacy budget consumption
- **Security metrics**: Vulnerability scan results

### Health Checks

All containers include health checks:

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# View health check logs
docker inspect --format='{{.State.Health}}' container_name
```

## Troubleshooting

### Common Issues

1. **Build failures**
   ```bash
   # Clear build cache
   docker system prune -a
   docker buildx prune -a
   ```

2. **GPU not available**
   ```bash
   # Check NVIDIA runtime
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
   ```

3. **Memory issues**
   ```bash
   # Increase Docker memory limit
   # Update Docker Desktop settings or systemd configuration
   ```

4. **Permission issues**
   ```bash
   # Fix volume permissions
   sudo chown -R $(id -u):$(id -g) ./data ./logs ./models
   ```

### Debug Commands

```bash
# Container debugging
docker-compose exec privacy-finetuner bash
docker logs -f privacy-finetuner

# Resource monitoring
docker stats
docker system df

# Network debugging
docker network ls
docker network inspect privacy-net
```

## CI/CD Integration

The build system integrates with CI/CD pipelines:

1. **Automated builds**: Triggered on code changes
2. **Security scanning**: Integrated vulnerability detection
3. **Multi-stage testing**: Unit, integration, security tests
4. **Automated deployment**: Staging and production pipelines
5. **Rollback capability**: Automated rollback on failures

See [CI/CD documentation](../workflows/README.md) for pipeline details.

## Best Practices

### Development

1. Use development containers for consistent environments
2. Mount source code for hot reload during development
3. Use separate databases for development and testing
4. Keep development secrets separate from production

### Production

1. Use multi-stage builds for minimal attack surface
2. Implement proper health checks and monitoring
3. Use secrets management for sensitive data
4. Implement proper backup and disaster recovery
5. Regular security scanning and updates

### Security

1. Never run containers as root in production
2. Use official base images from trusted sources
3. Regularly update base images and dependencies
4. Implement network segmentation and firewalls
5. Monitor and audit container activities

## Contributing

When adding new build features:

1. Update Dockerfiles following security best practices
2. Add appropriate health checks and labels
3. Update docker-compose configurations
4. Add security scanning configurations
5. Update documentation and examples
6. Test multi-architecture builds

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.
