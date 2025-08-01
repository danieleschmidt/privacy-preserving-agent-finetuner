#!/bin/bash

# Build script for Privacy-Preserving Agent Fine-Tuner
# Supports multiple build targets and security scanning

set -e

# Default values
BUILD_TARGET="production"
IMAGE_NAME="privacy-finetuner"
TAG="latest"
REGISTRY=""
PUSH=false
SCAN=false
CACHE=true
MULTI_ARCH=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --target TARGET     Build target (production, development, jupyter) [default: production]"
    echo "  -n, --name NAME         Image name [default: privacy-finetuner]"
    echo "  -g, --tag TAG           Image tag [default: latest]"
    echo "  -r, --registry REGISTRY Registry prefix (e.g., docker.io/username)"
    echo "  -p, --push              Push image to registry after building"
    echo "  -s, --scan              Run security scan after building"
    echo "  --no-cache              Disable Docker layer caching"
    echo "  --multi-arch            Build for multiple architectures (linux/amd64,linux/arm64)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -t production -g v1.0.0"
    echo "  $0 -t development -p"
    echo "  $0 -t jupyter --scan"
    echo "  $0 -r ghcr.io/username -p --multi-arch"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -g|--tag)
            TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -s|--scan)
            SCAN=true
            shift
            ;;
        --no-cache)
            CACHE=false
            shift
            ;;
        --multi-arch)
            MULTI_ARCH=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate build target
case $BUILD_TARGET in
    production|development|jupyter)
        ;;
    *)
        echo -e "${RED}Error: Invalid build target '$BUILD_TARGET'. Must be one of: production, development, jupyter${NC}"
        exit 1
        ;;
esac

# Construct full image name
if [ -n "$REGISTRY" ]; then
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${TAG}"
else
    FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"
fi

echo -e "${BLUE}🔨 Building Privacy-Preserving Agent Fine-Tuner${NC}"
echo -e "${BLUE}Target: ${BUILD_TARGET}${NC}"
echo -e "${BLUE}Image: ${FULL_IMAGE_NAME}${NC}"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must run from the root of the privacy-finetuner repository${NC}"
    exit 1
fi

# Determine Dockerfile
case $BUILD_TARGET in
    production)
        DOCKERFILE="Dockerfile"
        DOCKER_TARGET="production"
        ;;
    development)
        DOCKERFILE="Dockerfile.dev"
        DOCKER_TARGET=""
        ;;
    jupyter)
        DOCKERFILE="Dockerfile.jupyter"
        DOCKER_TARGET=""
        ;;
esac

# Build arguments
BUILD_ARGS=()

if [ "$CACHE" = false ]; then
    BUILD_ARGS+=(--no-cache)
fi

if [ -n "$DOCKER_TARGET" ]; then
    BUILD_ARGS+=(--target "$DOCKER_TARGET")
fi

# Add build metadata
BUILD_ARGS+=(--label "org.opencontainers.image.created=$(date -u +'%Y-%m-%dT%H:%M:%SZ')")
BUILD_ARGS+=(--label "org.opencontainers.image.revision=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')")
BUILD_ARGS+=(--label "org.opencontainers.image.version=${TAG}")

# Multi-architecture build
if [ "$MULTI_ARCH" = true ]; then
    echo -e "${YELLOW}🏗️  Building multi-architecture image...${NC}"
    
    # Check if buildx is available
    if ! docker buildx version &> /dev/null; then
        echo -e "${RED}Error: Docker buildx is required for multi-architecture builds${NC}"
        exit 1
    fi
    
    # Create buildx builder if it doesn't exist
    docker buildx create --name privacy-builder --use 2>/dev/null || docker buildx use privacy-builder
    
    BUILDX_ARGS=(
        buildx build
        --platform linux/amd64,linux/arm64
        "${BUILD_ARGS[@]}"
        -f "$DOCKERFILE"
        -t "$FULL_IMAGE_NAME"
    )
    
    if [ "$PUSH" = true ]; then
        BUILDX_ARGS+=(--push)
    else
        BUILDX_ARGS+=(--load)
    fi
    
    BUILDX_ARGS+=(.)
    
    echo -e "${BLUE}Running: docker ${BUILDX_ARGS[*]}${NC}"
    docker "${BUILDX_ARGS[@]}"
    
else
    # Single architecture build
    echo -e "${YELLOW}🏗️  Building image...${NC}"
    
    BUILD_CMD=(
        docker build
        "${BUILD_ARGS[@]}"
        -f "$DOCKERFILE"
        -t "$FULL_IMAGE_NAME"
        .
    )
    
    echo -e "${BLUE}Running: ${BUILD_CMD[*]}${NC}"
    "${BUILD_CMD[@]}"
fi

echo -e "${GREEN}✅ Build completed successfully${NC}"

# Security scanning
if [ "$SCAN" = true ]; then
    echo -e "${YELLOW}🔍 Running security scan...${NC}"
    
    # Check if trivy is available
    if command -v trivy &> /dev/null; then
        echo -e "${BLUE}Running Trivy security scan...${NC}"
        trivy image --format table --severity HIGH,CRITICAL "$FULL_IMAGE_NAME"
    else
        echo -e "${YELLOW}Warning: Trivy not found. Install it for comprehensive security scanning.${NC}"
        echo -e "${YELLOW}See: https://github.com/aquasecurity/trivy#installation${NC}"
    fi
    
    # Check if docker scout is available
    if docker scout version &> /dev/null; then
        echo -e "${BLUE}Running Docker Scout analysis...${NC}"
        docker scout cves "$FULL_IMAGE_NAME"
    fi
fi

# Push to registry
if [ "$PUSH" = true ] && [ "$MULTI_ARCH" = false ]; then
    echo -e "${YELLOW}📤 Pushing image to registry...${NC}"
    docker push "$FULL_IMAGE_NAME"
    echo -e "${GREEN}✅ Image pushed successfully${NC}"
fi

# Display image information
echo -e "${BLUE}📊 Image Information:${NC}"
docker images "$FULL_IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Display security recommendations
echo -e "${BLUE}🔒 Security Notes:${NC}"
echo "- Image runs as non-root user"
echo "- Minimal attack surface with distroless base"
echo "- Security scanning recommended before deployment"
echo "- Use specific tags instead of 'latest' in production"

echo -e "${GREEN}🎉 Build process completed!${NC}"

# Display next steps
echo -e "${BLUE}Next steps:${NC}"
case $BUILD_TARGET in
    production)
        echo "- Test the image: docker run --rm -p 8080:8080 $FULL_IMAGE_NAME"
        echo "- Check health: curl http://localhost:8080/health"
        ;;
    development)
        echo "- Start dev environment: docker-compose -f docker-compose.dev.yml up"
        echo "- Access container: docker exec -it privacy-finetuner-dev bash"
        ;;
    jupyter)
        echo "- Start Jupyter: docker run --rm -p 8888:8888 $FULL_IMAGE_NAME"
        echo "- Access notebook: http://localhost:8888 (token: dev-token-change-in-production)"
        ;;
esac