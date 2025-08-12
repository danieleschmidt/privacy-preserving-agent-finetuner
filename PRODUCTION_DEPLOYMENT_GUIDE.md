# Production Deployment Guide

## 🚀 Privacy-Preserving Agent Finetuner - Enterprise Deployment

This guide provides comprehensive instructions for deploying the Privacy-Preserving Agent Finetuner framework in production environments with enterprise-grade security, scalability, and compliance.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Preparation](#environment-preparation)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Provider Deployment](#cloud-provider-deployment)
- [Security Hardening](#security-hardening)
- [Monitoring & Observability](#monitoring--observability)
- [Backup & Recovery](#backup--recovery)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 8 cores (16 recommended)
- **Memory**: 32 GB RAM (64 GB recommended)
- **Storage**: 500 GB SSD (1 TB recommended)
- **Network**: 10 Gbps (for distributed training)

#### GPU Requirements (Optional but Recommended)
- **NVIDIA GPU**: RTX 4090, A100, H100, or equivalent
- **CUDA**: Version 11.8 or higher
- **GPU Memory**: 24 GB minimum (80 GB recommended for large models)
- **Driver**: NVIDIA driver 525.60.13 or higher

#### Software Dependencies
- **Operating System**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **Python**: 3.9, 3.10, 3.11, or 3.12
- **Docker**: 24.0+ (with BuildKit enabled)
- **Kubernetes**: 1.25+ (if using K8s deployment)
- **Git**: 2.30+ for source code management

### Network Requirements
- **Ingress**: HTTPS (443), HTTP (80) for web interfaces
- **API**: Custom ports 8080-8090 for microservices
- **Monitoring**: Port 9090 (Prometheus), 3000 (Grafana)
- **Database**: PostgreSQL (5432), Redis (6379)
- **Distributed Training**: Ports 29500-29510 for worker communication

## 🏗️ Environment Preparation

### 1. System Setup

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    build-essential \
    curl \
    wget \
    git \
    htop \
    nvtop \
    screen \
    tmux \
    jq

# Install Python and pip
sudo apt install -y python3.11 python3.11-dev python3.11-venv python3-pip

# Create system user for the application
sudo useradd -m -s /bin/bash privacy-ml
sudo usermod -aG docker privacy-ml
```

### 2. NVIDIA GPU Setup (if applicable)

```bash
# Install NVIDIA drivers
sudo apt install -y nvidia-driver-535

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### 3. Security Configuration

```bash
# Configure firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8080:8090/tcp

# Setup SSL/TLS certificates (using Let's Encrypt)
sudo apt install -y certbot
sudo certbot certonly --standalone -d your-domain.com

# Create certificate directory for Docker
sudo mkdir -p /etc/ssl/privacy-ml/
sudo cp /etc/letsencrypt/live/your-domain.com/* /etc/ssl/privacy-ml/
sudo chown -R privacy-ml:privacy-ml /etc/ssl/privacy-ml/
```

## 🐳 Docker Deployment

### 1. Build Production Images

Create a `docker-compose.production.yml` file:

```yaml
version: '3.8'

services:
  privacy-gateway:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    image: privacy-ml/gateway:${VERSION:-latest}
    container_name: privacy-gateway
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./models:/app/models
      - /etc/ssl/privacy-ml:/etc/ssl/privacy-ml:ro
    environment:
      - ENV=production
      - LOG_LEVEL=info
      - PRIVACY_BUDGET_LIMIT=10.0
      - SSL_CERT_PATH=/etc/ssl/privacy-ml/fullchain.pem
      - SSL_KEY_PATH=/etc/ssl/privacy-ml/privkey.pem
      - DATABASE_URL=postgresql://privacy_user:${DB_PASSWORD}@postgres:5432/privacy_db
      - REDIS_URL=redis://redis:6379/0
    secrets:
      - db_password
      - api_keys
      - model_encryption_key
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "https://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE

  ml-training-service:
    build:
      context: .
      dockerfile: Dockerfile
      target: training
    image: privacy-ml/training:${VERSION:-latest}
    container_name: ml-training-service
    restart: unless-stopped
    ports:
      - "8081:8081"
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./models:/app/models
      - ./datasets:/app/datasets:ro
      - ./checkpoints:/app/checkpoints
    environment:
      - ENV=production
      - BATCH_SIZE=32
      - PRIVACY_EPSILON=1.0
      - MAX_GRAD_NORM=1.0
      - NOISE_MULTIPLIER=0.5
      - CUDA_VISIBLE_DEVICES=0,1
    runtime: nvidia
    deploy:
      resources:
        limits:
          cpus: '16.0'
          memory: 64G
        reservations:
          cpus: '8.0'
          memory: 32G
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8081/metrics')"]
      interval: 60s
      timeout: 30s
      retries: 3

  compliance-monitor:
    build:
      context: .
      dockerfile: Dockerfile
      target: compliance
    image: privacy-ml/compliance:${VERSION:-latest}
    container_name: compliance-monitor
    restart: unless-stopped
    ports:
      - "8082:8082"
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./compliance:/app/compliance
    environment:
      - ENV=production
      - MONITORING_INTERVAL=300
      - ALERT_WEBHOOK=https://alerts.company.com/webhook
      - GDPR_ENABLED=true
      - CCPA_ENABLED=true
      - HIPAA_ENABLED=true
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  postgres:
    image: postgres:15-alpine
    container_name: privacy-postgres
    restart: unless-stopped
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql:ro
    environment:
      - POSTGRES_DB=privacy_db
      - POSTGRES_USER=privacy_user
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U privacy_user -d privacy_db"]
      interval: 30s
      timeout: 10s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: privacy-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./config/redis.conf:/usr/local/etc/redis/redis.conf:ro
    command: ["redis-server", "/usr/local/etc/redis/redis.conf"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    container_name: privacy-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    container_name: privacy-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SERVER_DOMAIN=your-domain.com
      - GF_SERVER_ROOT_URL=https://your-domain.com:3000/

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

secrets:
  db_password:
    file: ./secrets/db_password.txt
  api_keys:
    file: ./secrets/api_keys.json
  model_encryption_key:
    file: ./secrets/model_encryption_key.txt

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### 2. Production Docker Deployment

```bash
# Clone repository
git clone https://github.com/your-org/privacy-preserving-agent-finetuner
cd privacy-preserving-agent-finetuner

# Create necessary directories
mkdir -p logs models datasets checkpoints compliance secrets

# Create secrets
echo "your-secure-db-password" > secrets/db_password.txt
echo '{"api_key": "your-api-key"}' > secrets/api_keys.json
openssl rand -hex 32 > secrets/model_encryption_key.txt

# Set proper permissions
chmod 600 secrets/*

# Build and deploy
export VERSION=v1.0.0
docker-compose -f docker-compose.production.yml build
docker-compose -f docker-compose.production.yml up -d

# Verify deployment
docker-compose -f docker-compose.production.yml ps
docker-compose -f docker-compose.production.yml logs -f privacy-gateway
```

### 3. Health Check and Verification

```bash
# Check service health
curl -k https://localhost:8080/health
curl http://localhost:8081/metrics
curl http://localhost:8082/status

# Monitor logs
docker-compose -f docker-compose.production.yml logs -f

# Check resource usage
docker stats
```

## ☸️ Kubernetes Deployment

### 1. Namespace and RBAC Setup

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: privacy-ml
  labels:
    name: privacy-ml
    compliance: gdpr-ready
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: privacy-ml-sa
  namespace: privacy-ml
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: privacy-ml-cluster-role
rules:
- apiGroups: [""]
  resources: ["nodes", "pods", "services"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["autoscaling"]
  resources: ["horizontalpodautoscalers"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: privacy-ml-cluster-role-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: privacy-ml-cluster-role
subjects:
- kind: ServiceAccount
  name: privacy-ml-sa
  namespace: privacy-ml
```

### 2. ConfigMaps and Secrets

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: privacy-ml-config
  namespace: privacy-ml
data:
  privacy.yaml: |
    privacy:
      epsilon: 1.0
      delta: 1e-5
      max_grad_norm: 1.0
      noise_multiplier: 0.5
      accounting_mode: "rdp"
      
    security:
      enable_threat_detection: true
      alert_threshold: 0.7
      automated_response: true
      
    scaling:
      enable_auto_scaling: true
      min_replicas: 2
      max_replicas: 10
      target_cpu_utilization: 70
      
    compliance:
      frameworks: ["gdpr", "ccpa", "hipaa"]
      data_residency_required: true
      audit_logging: true
      
  redis.conf: |
    maxmemory 2gb
    maxmemory-policy allkeys-lru
    save 900 1
    save 300 10
    save 60 10000
    
---
apiVersion: v1
kind: Secret
metadata:
  name: privacy-ml-secrets
  namespace: privacy-ml
type: Opaque
data:
  db-password: <base64-encoded-password>
  api-keys: <base64-encoded-api-keys>
  model-encryption-key: <base64-encoded-encryption-key>
  ssl-cert: <base64-encoded-ssl-cert>
  ssl-key: <base64-encoded-ssl-key>
```

### 3. Persistent Volumes

```yaml
# persistent-volumes.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: privacy-ml-models-pv
spec:
  capacity:
    storage: 500Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: fast-ssd
  nfs:
    server: nfs-server.company.com
    path: /exports/privacy-ml/models
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: privacy-ml-models-pvc
  namespace: privacy-ml
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: privacy-ml-checkpoints-pvc
  namespace: privacy-ml
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Ti
  storageClassName: fast-ssd
```

### 4. Core Deployments

```yaml
# privacy-gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: privacy-gateway
  namespace: privacy-ml
  labels:
    app: privacy-gateway
    tier: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: privacy-gateway
  template:
    metadata:
      labels:
        app: privacy-gateway
        tier: api
    spec:
      serviceAccountName: privacy-ml-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: privacy-gateway
        image: privacy-ml/gateway:v1.0.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: https
        - containerPort: 8081
          name: metrics
        env:
        - name: ENV
          value: "production"
        - name: LOG_LEVEL
          value: "info"
        - name: DATABASE_URL
          value: "postgresql://privacy_user:$(DB_PASSWORD)@postgres:5432/privacy_db"
        - name: REDIS_URL
          value: "redis://redis:6379/0"
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: privacy-ml-secrets
              key: db-password
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: ssl-certs
          mountPath: /etc/ssl/privacy-ml
          readOnly: true
        - name: models
          mountPath: /app/models
        resources:
          limits:
            cpu: "4"
            memory: "8Gi"
          requests:
            cpu: "2"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
            scheme: HTTPS
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
            scheme: HTTPS
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 5
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
      volumes:
      - name: config
        configMap:
          name: privacy-ml-config
      - name: ssl-certs
        secret:
          secretName: privacy-ml-secrets
          items:
          - key: ssl-cert
            path: fullchain.pem
          - key: ssl-key
            path: privkey.pem
      - name: models
        persistentVolumeClaim:
          claimName: privacy-ml-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: privacy-gateway
  namespace: privacy-ml
  labels:
    app: privacy-gateway
spec:
  selector:
    app: privacy-gateway
  ports:
  - name: https
    port: 443
    targetPort: 8080
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 8081
    targetPort: 8081
  type: LoadBalancer
```

### 5. ML Training Service with GPU Support

```yaml
# ml-training-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-training-service
  namespace: privacy-ml
  labels:
    app: ml-training-service
    tier: compute
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-training-service
  template:
    metadata:
      labels:
        app: ml-training-service
        tier: compute
    spec:
      serviceAccountName: privacy-ml-sa
      nodeSelector:
        accelerator: nvidia-gpu
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: ml-training
        image: privacy-ml/training:v1.0.0
        ports:
        - containerPort: 8081
        env:
        - name: ENV
          value: "production"
        - name: BATCH_SIZE
          value: "32"
        - name: PRIVACY_EPSILON
          value: "1.0"
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1"
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: models
          mountPath: /app/models
        - name: checkpoints
          mountPath: /app/checkpoints
        - name: datasets
          mountPath: /app/datasets
          readOnly: true
        resources:
          limits:
            nvidia.com/gpu: 2
            cpu: "16"
            memory: "128Gi"
          requests:
            nvidia.com/gpu: 2
            cpu: "8"
            memory: "64Gi"
        livenessProbe:
          httpGet:
            path: /metrics
            port: 8081
          initialDelaySeconds: 60
          periodSeconds: 60
          timeoutSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8081
          initialDelaySeconds: 10
          periodSeconds: 30
          timeoutSeconds: 10
      volumes:
      - name: config
        configMap:
          name: privacy-ml-config
      - name: models
        persistentVolumeClaim:
          claimName: privacy-ml-models-pvc
      - name: checkpoints
        persistentVolumeClaim:
          claimName: privacy-ml-checkpoints-pvc
      - name: datasets
        hostPath:
          path: /mnt/datasets
          type: Directory
---
apiVersion: v1
kind: Service
metadata:
  name: ml-training-service
  namespace: privacy-ml
spec:
  selector:
    app: ml-training-service
  ports:
  - port: 8081
    targetPort: 8081
  type: ClusterIP
```

### 6. Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: privacy-gateway-hpa
  namespace: privacy-ml
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: privacy-gateway
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
```

### 7. Network Policies

```yaml
# network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: privacy-ml-network-policy
  namespace: privacy-ml
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: privacy-gateway
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 8081
  - from:
    - podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 8081
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

### 8. Deployment Commands

```bash
# Create namespace and RBAC
kubectl apply -f namespace.yaml

# Create ConfigMaps and Secrets
kubectl apply -f configmap.yaml

# Create PersistentVolumes
kubectl apply -f persistent-volumes.yaml

# Deploy core services
kubectl apply -f privacy-gateway-deployment.yaml
kubectl apply -f ml-training-deployment.yaml
kubectl apply -f compliance-monitor-deployment.yaml

# Create database services
kubectl apply -f postgres-deployment.yaml
kubectl apply -f redis-deployment.yaml

# Setup autoscaling
kubectl apply -f hpa.yaml

# Apply network policies
kubectl apply -f network-policies.yaml

# Create ingress
kubectl apply -f ingress.yaml

# Verify deployment
kubectl get all -n privacy-ml
kubectl describe hpa -n privacy-ml
kubectl logs -f deployment/privacy-gateway -n privacy-ml
```

## ☁️ Cloud Provider Deployment

### AWS Deployment with EKS

```bash
# Install AWS CLI and eksctl
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create EKS cluster
eksctl create cluster \
  --name privacy-ml-cluster \
  --version 1.25 \
  --region us-west-2 \
  --nodegroup-name privacy-ml-workers \
  --node-type m5.2xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed \
  --with-oidc \
  --ssh-access \
  --ssh-public-key ~/.ssh/id_rsa.pub

# Add GPU node group
eksctl create nodegroup \
  --cluster privacy-ml-cluster \
  --region us-west-2 \
  --name gpu-workers \
  --node-type p3.2xlarge \
  --nodes 2 \
  --nodes-min 0 \
  --nodes-max 5 \
  --node-ami-family AmazonLinux2 \
  --ssh-access \
  --ssh-public-key ~/.ssh/id_rsa.pub

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml

# Setup AWS Load Balancer Controller
eksctl utils associate-iam-oidc-provider --region us-west-2 --cluster privacy-ml-cluster --approve

curl -o iam_policy.json https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.4.7/docs/install/iam_policy.json

aws iam create-policy \
    --policy-name AWSLoadBalancerControllerIAMPolicy \
    --policy-document file://iam_policy.json

# Create EBS CSI driver
eksctl create addon --name aws-ebs-csi-driver --cluster privacy-ml-cluster --service-account-role-arn arn:aws:iam::ACCOUNT-ID:role/AmazonEKS_EBS_CSI_DriverRole --force

# Deploy application
kubectl apply -f kubernetes/
```

### Azure Deployment with AKS

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login and create resource group
az login
az group create --name privacy-ml-rg --location eastus

# Create AKS cluster
az aks create \
    --resource-group privacy-ml-rg \
    --name privacy-ml-cluster \
    --node-count 3 \
    --node-vm-size Standard_D8s_v3 \
    --enable-addons monitoring \
    --generate-ssh-keys \
    --enable-cluster-autoscaler \
    --min-count 1 \
    --max-count 10

# Add GPU node pool
az aks nodepool add \
    --resource-group privacy-ml-rg \
    --cluster-name privacy-ml-cluster \
    --name gpunodepool \
    --node-count 2 \
    --node-vm-size Standard_NC6s_v3 \
    --enable-cluster-autoscaler \
    --min-count 0 \
    --max-count 5

# Get credentials
az aks get-credentials --resource-group privacy-ml-rg --name privacy-ml-cluster

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml

# Deploy application
kubectl apply -f kubernetes/
```

### GCP Deployment with GKE

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Create GKE cluster
gcloud container clusters create privacy-ml-cluster \
    --zone us-central1-a \
    --machine-type n1-standard-8 \
    --num-nodes 3 \
    --enable-autoscaling \
    --min-nodes 1 \
    --max-nodes 10 \
    --enable-autorepair \
    --enable-autoupgrade

# Add GPU node pool
gcloud container node-pools create gpu-pool \
    --cluster privacy-ml-cluster \
    --zone us-central1-a \
    --machine-type n1-standard-4 \
    --accelerator type=nvidia-tesla-k80,count=1 \
    --num-nodes 2 \
    --enable-autoscaling \
    --min-nodes 0 \
    --max-nodes 5

# Get credentials
gcloud container clusters get-credentials privacy-ml-cluster --zone us-central1-a

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Deploy application
kubectl apply -f kubernetes/
```

## 🔒 Security Hardening

### 1. Container Security

```dockerfile
# Multi-stage security-hardened Dockerfile
FROM python:3.11-slim as base

# Create non-root user
RUN groupadd -r privacy-ml && useradd --no-log-init -r -g privacy-ml privacy-ml

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM base as production

# Copy Python dependencies
COPY --from=builder /root/.local /home/privacy-ml/.local

# Copy application
COPY --chown=privacy-ml:privacy-ml . /app
WORKDIR /app

# Set secure environment
ENV PATH=/home/privacy-ml/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Security settings
USER privacy-ml
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

CMD ["python", "-m", "privacy_finetuner.api.server"]
```

### 2. Secrets Management

```bash
# Using Kubernetes secrets with encryption at rest
kubectl create secret generic privacy-ml-secrets \
  --from-literal=db-password="$(openssl rand -base64 32)" \
  --from-literal=api-key="$(openssl rand -base64 64)" \
  --from-literal=encryption-key="$(openssl rand -hex 32)" \
  --dry-run=client -o yaml | kubectl apply -f -

# Enable encryption at rest
# Add to /etc/kubernetes/manifests/kube-apiserver.yaml
--encryption-provider-config=/etc/kubernetes/encryption-config.yaml
```

### 3. Network Security

```yaml
# security-policies.yaml
apiVersion: v1
kind: Pod
metadata:
  name: privacy-ml-pod
  annotations:
    seccomp.security.alpha.kubernetes.io/pod: runtime/default
    container.apparmor.security.beta.kubernetes.io/privacy-ml: runtime/default
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: privacy-ml
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
      runAsNonRoot: true
      runAsUser: 1000
```

### 4. RBAC Configuration

```yaml
# rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: privacy-ml
  name: privacy-ml-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: privacy-ml-role-binding
  namespace: privacy-ml
subjects:
- kind: ServiceAccount
  name: privacy-ml-sa
  namespace: privacy-ml
roleRef:
  kind: Role
  name: privacy-ml-role
  apiGroup: rbac.authorization.k8s.io
```

## 📊 Monitoring & Observability

### 1. Prometheus Configuration

```yaml
# prometheus-config.yaml
global:
  scrape_interval: 30s
  external_labels:
    cluster: 'privacy-ml-production'
    region: 'us-west-2'

rule_files:
  - "privacy_rules.yml"
  - "security_rules.yml"

scrape_configs:
  - job_name: 'privacy-gateway'
    static_configs:
    - targets: ['privacy-gateway:8081']
    metrics_path: /metrics
    scrape_interval: 15s
    
  - job_name: 'ml-training'
    static_configs:
    - targets: ['ml-training-service:8081']
    metrics_path: /metrics
    scrape_interval: 30s
    
  - job_name: 'compliance-monitor'
    static_configs:
    - targets: ['compliance-monitor:8082']
    metrics_path: /metrics
    scrape_interval: 60s

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
```

### 2. Alerting Rules

```yaml
# privacy-alerts.yml
groups:
- name: privacy.rules
  rules:
  - alert: PrivacyBudgetExhausted
    expr: privacy_epsilon_used / privacy_epsilon_total > 0.9
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Privacy budget nearly exhausted"
      description: "Privacy budget usage is at {{ $value }} which exceeds 90%"
      
  - alert: ThreatDetected
    expr: privacy_threat_detected > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Security threat detected"
      description: "Threat detected: {{ $labels.threat_type }}"
      
  - alert: TrainingFailure
    expr: up{job="ml-training"} == 0
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "ML training service is down"
      description: "ML training service has been down for more than 2 minutes"

- name: performance.rules
  rules:
  - alert: HighMemoryUsage
    expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is above 90% for {{ $labels.container }}"
      
  - alert: HighCPUUsage
    expr: (rate(container_cpu_usage_seconds_total[5m])) > 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is above 80% for {{ $labels.container }}"
```

### 3. Grafana Dashboards

```json
{
  "dashboard": {
    "title": "Privacy-Preserving ML Dashboard",
    "panels": [
      {
        "title": "Privacy Budget Usage",
        "type": "gauge",
        "targets": [
          {
            "expr": "privacy_epsilon_used / privacy_epsilon_total",
            "legendFormat": "Budget Usage"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 0.7},
                {"color": "red", "value": 0.9}
              ]
            }
          }
        }
      },
      {
        "title": "Training Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(training_samples_processed_total[5m])",
            "legendFormat": "Samples/sec"
          },
          {
            "expr": "training_accuracy",
            "legendFormat": "Accuracy"
          }
        ]
      },
      {
        "title": "Security Threats",
        "type": "table",
        "targets": [
          {
            "expr": "privacy_threats_detected_total",
            "legendFormat": "{{ threat_type }}"
          }
        ]
      }
    ]
  }
}
```

### 4. Logging Configuration

```yaml
# fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: privacy-ml
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      read_from_head true
      <parse>
        @type json
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>
    
    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>
    
    <filter kubernetes.**>
      @type grep
      <regexp>
        key $.kubernetes.namespace_name
        pattern ^privacy-ml$
      </regexp>
    </filter>
    
    <filter kubernetes.**>
      @type privacy_redaction
      patterns ["SSN", "CREDIT_CARD", "EMAIL", "PHONE"]
    </filter>
    
    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name privacy-ml-logs
    </match>
```

## 💾 Backup & Recovery

### 1. Database Backup

```bash
#!/bin/bash
# backup-database.sh

# Configuration
DB_HOST="postgres.privacy-ml.svc.cluster.local"
DB_NAME="privacy_db"
DB_USER="privacy_user"
BACKUP_DIR="/backups/postgres"
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR

# Create timestamped backup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/privacy_db_$TIMESTAMP.sql"

# Perform backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Upload to cloud storage (AWS S3 example)
aws s3 cp $BACKUP_FILE.gz s3://privacy-ml-backups/postgres/

# Clean up old backups
find $BACKUP_DIR -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete

# Log backup completion
echo "$(date): Database backup completed - $BACKUP_FILE.gz" >> /var/log/backup.log
```

### 2. Model Checkpoint Backup

```bash
#!/bin/bash
# backup-checkpoints.sh

# Configuration
CHECKPOINT_DIR="/app/checkpoints"
BACKUP_BUCKET="s3://privacy-ml-backups/checkpoints"
RETENTION_DAYS=7

# Sync checkpoints to cloud storage
aws s3 sync $CHECKPOINT_DIR $BACKUP_BUCKET --delete

# Archive old checkpoints
find $CHECKPOINT_DIR -name "*.pkl" -mtime +$RETENTION_DAYS -exec rm {} \;

# Log backup completion
echo "$(date): Checkpoint backup completed" >> /var/log/backup.log
```

### 3. Disaster Recovery Plan

```yaml
# disaster-recovery.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: disaster-recovery-plan
  namespace: privacy-ml
data:
  recovery-steps.md: |
    # Disaster Recovery Procedures
    
    ## Database Recovery
    1. Restore from latest backup:
       ```
       psql -h postgres -U privacy_user -d privacy_db < backup.sql
       ```
    
    ## Model Recovery
    1. Download latest checkpoint:
       ```
       aws s3 sync s3://privacy-ml-backups/checkpoints /app/checkpoints
       ```
    
    ## Service Recovery
    1. Redeploy services:
       ```
       kubectl apply -f kubernetes/
       ```
    
    ## Verification
    1. Health checks:
       ```
       curl -k https://privacy-gateway/health
       ```
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Pod Startup Issues

```bash
# Check pod status
kubectl get pods -n privacy-ml

# Describe failing pod
kubectl describe pod <pod-name> -n privacy-ml

# Check logs
kubectl logs <pod-name> -n privacy-ml --previous

# Common fixes:
# - Check resource limits
# - Verify secrets and configmaps
# - Check persistent volume claims
```

#### 2. GPU Not Detected

```bash
# Verify NVIDIA device plugin
kubectl get daemonset -n kube-system nvidia-device-plugin-daemonset

# Check GPU nodes
kubectl get nodes -o json | jq '.items[] | select(.status.allocatable."nvidia.com/gpu" != null) | {name: .metadata.name, gpu: .status.allocatable."nvidia.com/gpu"}'

# Verify GPU in pod
kubectl exec -it <pod-name> -n privacy-ml -- nvidia-smi
```

#### 3. Privacy Budget Issues

```bash
# Check privacy metrics
kubectl exec -it privacy-gateway-xxx -n privacy-ml -- \
  curl http://localhost:8081/metrics | grep privacy

# Reset privacy budget (emergency)
kubectl exec -it privacy-gateway-xxx -n privacy-ml -- \
  python -c "from privacy_finetuner.core.privacy_config import reset_privacy_budget; reset_privacy_budget()"
```

#### 4. Performance Issues

```bash
# Check resource usage
kubectl top pods -n privacy-ml
kubectl top nodes

# Check HPA status
kubectl describe hpa -n privacy-ml

# Scale manually if needed
kubectl scale deployment privacy-gateway --replicas=5 -n privacy-ml
```

### Monitoring Commands

```bash
# Real-time pod monitoring
watch kubectl get pods -n privacy-ml

# Continuous log streaming
kubectl logs -f deployment/privacy-gateway -n privacy-ml

# Resource monitoring
kubectl top pods -n privacy-ml --containers

# Service endpoint testing
kubectl run test-pod --image=curlimages/curl -i --tty --rm -- \
  curl http://privacy-gateway.privacy-ml.svc.cluster.local/health
```

### Emergency Procedures

```bash
# Emergency shutdown
kubectl scale deployment --all --replicas=0 -n privacy-ml

# Emergency privacy budget reset
kubectl create job privacy-budget-reset --image=privacy-ml/gateway:latest -- \
  python -c "from privacy_finetuner.core.privacy_config import emergency_reset; emergency_reset()"

# Rollback deployment
kubectl rollout undo deployment/privacy-gateway -n privacy-ml

# Emergency backup
kubectl exec -it postgres-xxx -n privacy-ml -- \
  pg_dump -U privacy_user privacy_db > /tmp/emergency_backup.sql
```

---

## 📞 Support and Maintenance

For production support:
- **Email**: privacy-ai@your-org.com
- **Slack**: #privacy-ml-ops
- **On-call**: +1-xxx-xxx-xxxx
- **Documentation**: https://docs.your-org.com/privacy-finetuner

Regular maintenance tasks:
- Weekly: Review security alerts and update dependencies
- Monthly: Performance optimization and capacity planning
- Quarterly: Security audit and compliance review
- Annually: Disaster recovery testing

This guide provides comprehensive deployment instructions for enterprise production environments. Always test deployments in staging environments first and ensure all security requirements are met before production deployment.