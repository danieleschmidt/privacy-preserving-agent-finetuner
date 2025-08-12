#!/bin/bash

# Privacy Finetuner Deployment Script
# This script automates the deployment of Privacy Finetuner on Kubernetes

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
NAMESPACE=${NAMESPACE:-"privacy-finetuner"}
ENVIRONMENT=${ENVIRONMENT:-"production"}
DRY_RUN=${DRY_RUN:-"false"}
SKIP_DEPENDENCIES=${SKIP_DEPENDENCIES:-"false"}
VALUES_FILE=${VALUES_FILE:-""}

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if kubectl is installed and configured
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        print_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we can connect to the cluster
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to install dependencies
install_dependencies() {
    if [ "$SKIP_DEPENDENCIES" = "true" ]; then
        print_warning "Skipping dependency installation"
        return
    fi
    
    print_status "Installing dependencies..."
    
    # Add Helm repositories
    print_status "Adding Helm repositories..."
    helm repo add bitnami https://charts.bitnami.com/bitnami || true
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true
    helm repo add grafana https://grafana.github.io/helm-charts || true
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx || true
    helm repo add nvidia https://nvidia.github.io/gpu-operator || true
    helm repo update
    
    print_success "Helm repositories added"
    
    # Install NVIDIA GPU Operator (if not present)
    if ! kubectl get namespace gpu-operator &> /dev/null; then
        print_status "Installing NVIDIA GPU Operator..."
        helm install gpu-operator nvidia/gpu-operator \
            --namespace gpu-operator \
            --create-namespace \
            --wait
        print_success "NVIDIA GPU Operator installed"
    else
        print_warning "NVIDIA GPU Operator already exists"
    fi
    
    # Install NGINX Ingress Controller (if not present)
    if ! kubectl get namespace ingress-nginx &> /dev/null; then
        print_status "Installing NGINX Ingress Controller..."
        helm install ingress-nginx ingress-nginx/ingress-nginx \
            --namespace ingress-nginx \
            --create-namespace \
            --wait
        print_success "NGINX Ingress Controller installed"
    else
        print_warning "NGINX Ingress Controller already exists"
    fi
    
    # Install cert-manager (if not present)
    if ! kubectl get namespace cert-manager &> /dev/null; then
        print_status "Installing cert-manager..."
        kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
        kubectl wait --for=condition=available --timeout=300s deployment/cert-manager -n cert-manager
        print_success "cert-manager installed"
    else
        print_warning "cert-manager already exists"
    fi
    
    # Install Prometheus stack (if monitoring is enabled and not present)
    if ! kubectl get namespace monitoring &> /dev/null; then
        print_status "Installing Prometheus monitoring stack..."
        helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
            --namespace monitoring \
            --create-namespace \
            --wait
        print_success "Prometheus monitoring stack installed"
    else
        print_warning "Monitoring stack already exists"
    fi
}

# Function to create namespace
create_namespace() {
    print_status "Creating namespace $NAMESPACE..."
    
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        print_warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace $NAMESPACE
        print_success "Namespace $NAMESPACE created"
    fi
    
    # Label namespace for monitoring
    kubectl label namespace $NAMESPACE name=$NAMESPACE --overwrite
}

# Function to deploy using raw manifests
deploy_manifests() {
    print_status "Deploying using raw Kubernetes manifests..."
    
    # Deploy in order
    kubectl apply -f namespace.yaml
    kubectl apply -f rbac.yaml
    kubectl apply -f configmap.yaml
    kubectl apply -f secrets.yaml
    kubectl apply -f persistent-volumes.yaml
    
    # Wait for PVCs to be bound
    print_status "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=Bound pvc --all -n $NAMESPACE --timeout=300s
    
    # Deploy applications
    kubectl apply -f deployment.yaml
    kubectl apply -f services.yaml
    kubectl apply -f hpa.yaml
    kubectl apply -f ingress.yaml
    kubectl apply -f monitoring.yaml
    
    print_success "Manifests deployed successfully"
}

# Function to deploy using Helm
deploy_helm() {
    print_status "Deploying using Helm..."
    
    local helm_args="--namespace $NAMESPACE --create-namespace"
    
    # Add values file if specified
    if [ -n "$VALUES_FILE" ]; then
        if [ -f "$VALUES_FILE" ]; then
            helm_args="$helm_args --values $VALUES_FILE"
        else
            print_error "Values file $VALUES_FILE not found"
            exit 1
        fi
    fi
    
    # Add environment-specific settings
    case $ENVIRONMENT in
        "development")
            helm_args="$helm_args --set workers.replicaCount=1 --set autoscaling.hpa.enabled=false"
            helm_args="$helm_args --set persistence.data.size=100Gi --set config.application.environment=development"
            ;;
        "staging")
            helm_args="$helm_args --set workers.replicaCount=2 --set autoscaling.hpa.workers.maxReplicas=5"
            helm_args="$helm_args --set config.application.environment=staging"
            ;;
        "production")
            helm_args="$helm_args --set workers.replicaCount=3 --set autoscaling.hpa.workers.maxReplicas=10"
            helm_args="$helm_args --set config.application.environment=production"
            helm_args="$helm_args --set ingress.enabled=true --set monitoring.serviceMonitor.enabled=true"
            ;;
    esac
    
    # Dry run if requested
    if [ "$DRY_RUN" = "true" ]; then
        helm_args="$helm_args --dry-run --debug"
        print_warning "Performing dry run..."
    fi
    
    # Deploy with Helm
    helm upgrade --install privacy-finetuner ./helm/privacy-finetuner $helm_args
    
    if [ "$DRY_RUN" = "false" ]; then
        print_success "Helm deployment completed"
    else
        print_success "Dry run completed"
    fi
}

# Function to wait for deployment
wait_for_deployment() {
    if [ "$DRY_RUN" = "true" ]; then
        return
    fi
    
    print_status "Waiting for deployment to be ready..."
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=available --timeout=600s deployment --all -n $NAMESPACE
    
    # Check pod status
    kubectl get pods -n $NAMESPACE
    
    print_success "Deployment is ready!"
}

# Function to run post-deployment tests
run_tests() {
    if [ "$DRY_RUN" = "true" ]; then
        return
    fi
    
    print_status "Running post-deployment tests..."
    
    # Test Helm deployment if using Helm
    if [ "$USE_HELM" = "true" ]; then
        helm test privacy-finetuner -n $NAMESPACE || true
    fi
    
    # Check service connectivity
    print_status "Checking service connectivity..."
    kubectl get services -n $NAMESPACE
    
    # Check HPA status
    print_status "Checking HPA status..."
    kubectl get hpa -n $NAMESPACE
    
    print_success "Post-deployment tests completed"
}

# Function to show access information
show_access_info() {
    if [ "$DRY_RUN" = "true" ]; then
        return
    fi
    
    print_success "=== Privacy Finetuner Deployment Complete ==="
    echo
    print_status "Access Information:"
    echo
    
    # Get service information
    echo "Services:"
    kubectl get services -n $NAMESPACE
    echo
    
    # Get ingress information
    if kubectl get ingress -n $NAMESPACE &> /dev/null; then
        echo "Ingress:"
        kubectl get ingress -n $NAMESPACE
        echo
    fi
    
    # Show monitoring access
    echo "Monitoring:"
    echo "  Grafana: kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80"
    echo "  Prometheus: kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090"
    echo
    
    # Show useful commands
    echo "Useful Commands:"
    echo "  View pods: kubectl get pods -n $NAMESPACE"
    echo "  View logs: kubectl logs -f deployment/privacy-finetuner-master -n $NAMESPACE"
    echo "  Scale workers: kubectl scale deployment privacy-finetuner-worker --replicas=5 -n $NAMESPACE"
    echo "  Port-forward: kubectl port-forward -n $NAMESPACE svc/privacy-finetuner-master 8080:8080"
    echo
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -n, --namespace NAMESPACE      Target namespace (default: privacy-finetuner)"
    echo "  -e, --environment ENV          Environment (development|staging|production, default: production)"
    echo "  -f, --values-file FILE         Path to Helm values file"
    echo "  -m, --manifests               Use raw Kubernetes manifests instead of Helm"
    echo "  -d, --dry-run                 Perform dry run without actual deployment"
    echo "  -s, --skip-dependencies       Skip dependency installation"
    echo "  -h, --help                    Show this help message"
    echo
    echo "Examples:"
    echo "  $0                                           # Deploy to production with defaults"
    echo "  $0 -e development -n dev                     # Deploy to development environment"
    echo "  $0 -f custom-values.yaml                     # Deploy with custom values"
    echo "  $0 -m                                        # Deploy using raw manifests"
    echo "  $0 -d                                        # Dry run to see what would be deployed"
}

# Parse command line arguments
USE_HELM="true"

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -f|--values-file)
            VALUES_FILE="$2"
            shift 2
            ;;
        -m|--manifests)
            USE_HELM="false"
            shift
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        -s|--skip-dependencies)
            SKIP_DEPENDENCIES="true"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main deployment flow
main() {
    print_status "Starting Privacy Finetuner deployment..."
    print_status "Environment: $ENVIRONMENT"
    print_status "Namespace: $NAMESPACE"
    print_status "Deployment method: $([ "$USE_HELM" = "true" ] && echo "Helm" || echo "Raw manifests")"
    
    # Run deployment steps
    check_prerequisites
    install_dependencies
    create_namespace
    
    if [ "$USE_HELM" = "true" ]; then
        deploy_helm
    else
        deploy_manifests
    fi
    
    wait_for_deployment
    run_tests
    show_access_info
    
    print_success "Privacy Finetuner deployment completed successfully! 🚀"
}

# Run main function
main