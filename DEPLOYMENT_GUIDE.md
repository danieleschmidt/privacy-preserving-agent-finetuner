# Privacy-Preserving ML Framework - Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized deployment)
- Kubernetes cluster (optional, for orchestrated deployment)
- Cloud provider account (AWS/Azure/GCP, optional)

### Basic Installation
```bash
# Clone the repository
git clone <repository-url>
cd privacy-preserving-agent-finetuner

# Install dependencies
pip install -r requirements.txt

# Run basic demo
python examples/generation1_demo.py
```

### Docker Deployment
```bash
# Build container
docker build -t privacy-ml-framework .

# Run with basic configuration
docker run -p 8080:8080 privacy-ml-framework

# Run with advanced features
docker run -e ENABLE_SCALING=true -e ENABLE_MONITORING=true privacy-ml-framework
```

## 🏗️ Architecture Overview

### System Components
```
┌─────────────────────────────────────────────────────────────┐
│                    Privacy-Preserving ML Framework          │
├─────────────────────────────────────────────────────────────┤
│  Research Layer    │ Novel algorithms, benchmarking        │
│  Security Layer    │ Threat detection, recovery            │ 
│  Scaling Layer     │ Performance optimization, auto-scale  │
│  Quality Layer     │ Testing, validation, compliance       │
│  Global Layer      │ I18n, compliance, deployment          │
├─────────────────────────────────────────────────────────────┤
│                    Core Privacy Framework                   │
└─────────────────────────────────────────────────────────────┘
```

### Service Architecture
```
Internet/API Gateway
        │
    ┌───▼───┐      ┌──────────────┐      ┌─────────────────┐
    │Privacy│      │   ML Training │      │  Compliance     │
    │Gateway│◄────►│   Service     │◄────►│  Monitor        │
    └───┬───┘      └──────────────┘      └─────────────────┘
        │                    │                      │
    ┌───▼───┐      ┌─────────▼──────┐      ┌───────▼──────────┐
    │Auto   │      │ Performance     │      │  Threat          │
    │Scaler │      │ Optimizer       │      │  Detector        │
    └───────┘      └────────────────┘      └──────────────────┘
```

## 🌍 Multi-Region Deployment

### Regional Configuration

#### Europe (GDPR Compliance)
```yaml
# europe-config.yaml
region: eu-west-1
compliance_frameworks:
  - GDPR
data_residency_required: true
privacy_settings:
  consent_management: explicit
  data_retention_days: 365
  subject_rights_enabled: true
localization:
  primary_locale: de_DE
  supported_locales: [en_GB, fr_FR, it_IT, es_ES]
```

#### United States (CCPA/HIPAA)
```yaml
# us-config.yaml  
region: us-east-1
compliance_frameworks:
  - CCPA
  - HIPAA  # for healthcare deployments
data_residency_required: false
privacy_settings:
  opt_out_mechanism: enabled
  sale_disclosure: required
  minor_protection: enhanced
localization:
  primary_locale: en_US
  supported_locales: [es_MX, fr_CA]
```

#### Asia-Pacific (PDPA)
```yaml
# asia-config.yaml
region: asia-southeast-1
compliance_frameworks:
  - PDPA_Singapore
data_residency_required: true
privacy_settings:
  notification_requirements: strict
  cross_border_restrictions: enabled
localization:
  primary_locale: en_US
  supported_locales: [zh_CN, ja_JP, ko_KR]
```

### Deployment Commands
```bash
# Deploy to Europe
./deploy.sh --region eu-west-1 --config europe-config.yaml

# Deploy to United States
./deploy.sh --region us-east-1 --config us-config.yaml

# Deploy to Asia-Pacific
./deploy.sh --region asia-southeast-1 --config asia-config.yaml
```

## ☸️ Kubernetes Deployment

### Basic Kubernetes Manifests

#### Privacy Gateway Deployment
```yaml
# privacy-gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: privacy-gateway
  labels:
    app: privacy-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: privacy-gateway
  template:
    metadata:
      labels:
        app: privacy-gateway
    spec:
      containers:
      - name: privacy-gateway
        image: privacy-ml/gateway:latest
        ports:
        - containerPort: 8080
        env:
        - name: PRIVACY_BUDGET_LIMIT
          value: "10.0"
        - name: ENABLE_MONITORING
          value: "true"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: privacy-gateway-service
spec:
  selector:
    app: privacy-gateway
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

#### ML Training Service
```yaml
# ml-training-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-training-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-training-service
  template:
    metadata:
      labels:
        app: ml-training-service
    spec:
      containers:
      - name: ml-training
        image: privacy-ml/training:latest
        ports:
        - containerPort: 8081
        env:
        - name: BATCH_SIZE
          value: "32"
        - name: PRIVACY_EPSILON
          value: "1.0"
        resources:
          requests:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "8000m"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
```

### Horizontal Pod Autoscaler
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: privacy-gateway-hpa
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
```

### ConfigMap for Compliance
```yaml
# compliance-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: compliance-config
data:
  gdpr.yaml: |
    framework: GDPR
    data_retention_days: 365
    consent_required: true
    subject_rights:
      - access
      - rectification
      - erasure
      - portability
      - restriction
      - objection
  ccpa.yaml: |
    framework: CCPA
    data_retention_days: 730
    opt_out_required: true
    sale_disclosure: true
    consumer_rights:
      - know
      - delete
      - opt_out_sale
      - non_discrimination
```

## 🏢 Enterprise Deployment

### High Availability Setup

#### Load Balancer Configuration
```yaml
# nginx-lb.conf
upstream privacy_gateway {
    server privacy-gateway-1:8080 weight=3;
    server privacy-gateway-2:8080 weight=3;
    server privacy-gateway-3:8080 weight=2;
    server privacy-gateway-4:8080 weight=2 backup;
}

server {
    listen 443 ssl http2;
    server_name api.privacy-ml.company.com;
    
    ssl_certificate /etc/ssl/certs/privacy-ml.crt;
    ssl_certificate_key /etc/ssl/private/privacy-ml.key;
    
    location / {
        proxy_pass http://privacy_gateway;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /health {
        access_log off;
        proxy_pass http://privacy_gateway;
        proxy_set_header Host $host;
    }
}
```

#### Database Configuration
```yaml
# database.yaml
apiVersion: v1
kind: Secret
metadata:
  name: database-credentials
type: Opaque
data:
  username: <base64-encoded-username>
  password: <base64-encoded-password>
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: privacy-database
spec:
  serviceName: privacy-database-service
  replicas: 3
  selector:
    matchLabels:
      app: privacy-database
  template:
    metadata:
      labels:
        app: privacy-database
    spec:
      containers:
      - name: postgres
        image: postgres:13-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: privacy_ml
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

### Monitoring & Observability

#### Prometheus Configuration
```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'privacy-gateway'
      static_configs:
      - targets: ['privacy-gateway-service:8080']
      metrics_path: /metrics
    - job_name: 'ml-training'
      static_configs:
      - targets: ['ml-training-service:8081']
      metrics_path: /metrics
    - job_name: 'compliance-monitor'
      static_configs:
      - targets: ['compliance-service:8082']
      metrics_path: /metrics
    rule_files:
    - "/etc/prometheus/privacy-ml-rules.yml"
    alerting:
      alertmanagers:
      - static_configs:
        - targets: ['alertmanager:9093']
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Privacy-Preserving ML Framework",
    "panels": [
      {
        "title": "Privacy Budget Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "privacy_budget_used / privacy_budget_total * 100",
            "legendFormat": "Budget Utilization %"
          }
        ]
      },
      {
        "title": "Compliance Violations",
        "type": "singlestat",
        "targets": [
          {
            "expr": "compliance_violations_total",
            "legendFormat": "Active Violations"
          }
        ]
      },
      {
        "title": "Threat Detection Rate",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(threats_detected_total[5m])",
            "legendFormat": "Threats/sec"
          }
        ]
      }
    ]
  }
}
```

## 🔧 Configuration Management

### Environment Variables
```bash
# Core Configuration
export PRIVACY_EPSILON=1.0
export PRIVACY_DELTA=1e-5
export BATCH_SIZE=32
export LEARNING_RATE=5e-5

# Security Configuration
export ENABLE_THREAT_DETECTION=true
export THREAT_DETECTION_INTERVAL=30
export AUTO_REMEDIATION=true
export SECURITY_ALERT_WEBHOOK=https://alerts.company.com/webhook

# Scaling Configuration
export ENABLE_AUTO_SCALING=true
export MIN_REPLICAS=2
export MAX_REPLICAS=20
export SCALE_UP_THRESHOLD=75
export SCALE_DOWN_THRESHOLD=25

# Compliance Configuration
export COMPLIANCE_FRAMEWORK=GDPR,CCPA
export DATA_RETENTION_DAYS=365
export ENABLE_CONSENT_MANAGEMENT=true
export PRIVACY_OFFICER_EMAIL=privacy@company.com

# Internationalization
export DEFAULT_LOCALE=en_US
export SUPPORTED_LOCALES=en_US,de_DE,fr_FR,ja_JP,zh_CN
export ENABLE_RTL_SUPPORT=true

# Database Configuration
export DATABASE_URL=postgresql://user:pass@localhost:5432/privacy_ml
export DATABASE_POOL_SIZE=20
export DATABASE_ENCRYPTION=true

# Monitoring Configuration
export ENABLE_METRICS=true
export METRICS_PORT=9090
export LOG_LEVEL=INFO
export STRUCTURED_LOGGING=true
```

### Configuration Files

#### Main Configuration (`config/production.yaml`)
```yaml
privacy:
  epsilon: 1.0
  delta: 1e-5
  budget_management: adaptive
  noise_mechanism: gaussian

security:
  threat_detection:
    enabled: true
    interval_seconds: 30
    auto_remediation: true
  encryption:
    at_rest: true
    in_transit: true
    key_rotation: weekly

scaling:
  auto_scaling:
    enabled: true
    min_replicas: 3
    max_replicas: 20
    cpu_threshold: 75
    memory_threshold: 80
  performance_optimization:
    enabled: true
    optimization_interval: 300

compliance:
  frameworks: [GDPR, CCPA, HIPAA]
  data_retention_days: 365
  consent_management: explicit
  audit_logging: true

internationalization:
  default_locale: en_US
  fallback_locale: en_US
  supported_locales:
    - en_US
    - de_DE
    - fr_FR
    - ja_JP
    - zh_CN
    - ar_SA

deployment:
  strategy: canary
  health_check_timeout: 300
  rollback_enabled: true
  multi_region: true
```

## 🔒 Security Configuration

### SSL/TLS Setup
```bash
# Generate certificates
openssl req -x509 -newkey rsa:4096 -keyout privacy-ml.key -out privacy-ml.crt -days 365 -nodes

# Create Kubernetes secret
kubectl create secret tls privacy-ml-tls --cert=privacy-ml.crt --key=privacy-ml.key
```

### Network Policies
```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: privacy-ml-network-policy
spec:
  podSelector:
    matchLabels:
      app: privacy-gateway
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: api-gateway
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: ml-training-service
    ports:
    - protocol: TCP
      port: 8081
  - to:
    - podSelector:
        matchLabels:
          app: privacy-database
    ports:
    - protocol: TCP
      port: 5432
```

### RBAC Configuration
```yaml
# rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: privacy-ml-service-account
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: privacy-ml-cluster-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "update", "patch"]
- apiGroups: ["autoscaling"]
  resources: ["horizontalpodautoscalers"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
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
  name: privacy-ml-service-account
  namespace: default
```

## 📊 Monitoring & Alerting

### Alert Rules
```yaml
# privacy-ml-alerts.yml
groups:
- name: privacy-ml
  rules:
  - alert: HighPrivacyBudgetUsage
    expr: privacy_budget_used / privacy_budget_total > 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Privacy budget usage is high"
      description: "Privacy budget usage is {{ $value }}%"

  - alert: ComplianceViolation
    expr: compliance_violations_total > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Compliance violation detected"
      description: "{{ $value }} compliance violations detected"

  - alert: ThreatDetected
    expr: increase(threats_detected_total[5m]) > 0
    for: 0m
    labels:
      severity: warning
    annotations:
      summary: "Security threat detected"
      description: "{{ $value }} threats detected in the last 5 minutes"

  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service is down"
      description: "{{ $labels.instance }} has been down for more than 1 minute"
```

### Health Checks
```python
# health_check.py
from flask import Flask, jsonify
import psutil
import requests

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    })

@app.route('/ready')
def readiness_check():
    """Readiness check with dependencies"""
    checks = {
        "database": check_database(),
        "privacy_budget": check_privacy_budget(),
        "compliance": check_compliance_status()
    }
    
    all_ready = all(checks.values())
    
    return jsonify({
        "ready": all_ready,
        "checks": checks
    }), 200 if all_ready else 503

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return generate_metrics()
```

## 🚀 Deployment Scripts

### Basic Deployment Script
```bash
#!/bin/bash
# deploy.sh

set -e

# Configuration
NAMESPACE=${NAMESPACE:-default}
ENVIRONMENT=${ENVIRONMENT:-production}
REGION=${REGION:-us-east-1}
CONFIG_FILE=${CONFIG_FILE:-config/production.yaml}

echo "🚀 Deploying Privacy-Preserving ML Framework"
echo "   Environment: $ENVIRONMENT"
echo "   Region: $REGION"
echo "   Namespace: $NAMESPACE"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply configuration
kubectl create configmap app-config --from-file=$CONFIG_FILE -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy database
kubectl apply -f k8s/database.yaml -n $NAMESPACE

# Wait for database to be ready
kubectl wait --for=condition=ready pod -l app=privacy-database -n $NAMESPACE --timeout=300s

# Deploy core services
kubectl apply -f k8s/privacy-gateway-deployment.yaml -n $NAMESPACE
kubectl apply -f k8s/ml-training-deployment.yaml -n $NAMESPACE
kubectl apply -f k8s/compliance-service-deployment.yaml -n $NAMESPACE

# Deploy autoscaling
kubectl apply -f k8s/hpa.yaml -n $NAMESPACE

# Deploy ingress
kubectl apply -f k8s/ingress.yaml -n $NAMESPACE

# Wait for deployments
kubectl wait --for=condition=available deployment -l tier=privacy-ml -n $NAMESPACE --timeout=600s

# Run health checks
echo "🔍 Running health checks..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE

echo "✅ Deployment completed successfully!"
echo "🌐 Access the application at: https://$(kubectl get ingress privacy-ml-ingress -n $NAMESPACE -o jsonpath='{.spec.rules[0].host}')"
```

### Multi-Region Deployment
```bash
#!/bin/bash
# multi-region-deploy.sh

REGIONS=("us-east-1" "eu-west-1" "asia-southeast-1")
CONTEXTS=("us-cluster" "eu-cluster" "asia-cluster")

for i in "${!REGIONS[@]}"; do
    REGION="${REGIONS[$i]}"
    CONTEXT="${CONTEXTS[$i]}"
    
    echo "🌍 Deploying to region: $REGION"
    
    # Switch context
    kubectl config use-context $CONTEXT
    
    # Deploy with region-specific configuration
    REGION=$REGION CONFIG_FILE=config/$REGION.yaml ./deploy.sh
    
    # Verify deployment
    kubectl get deployments -l region=$REGION
done

echo "✅ Multi-region deployment completed!"
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy Privacy-ML Framework

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ --cov=privacy_finetuner --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: security-scan-results.sarif

  deploy-staging:
    needs: [test, security-scan]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        kubectl config use-context staging-cluster
        ENVIRONMENT=staging ./deploy.sh

  deploy-production:
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        kubectl config use-context production-cluster
        ENVIRONMENT=production ./deploy.sh
```

## 📈 Scaling Guidelines

### Horizontal Scaling
```yaml
# Recommended scaling parameters
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: privacy-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: privacy-gateway
  minReplicas: 3          # Minimum for high availability
  maxReplicas: 50         # Scale to handle high load
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
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Vertical Scaling
```yaml
# VPA configuration
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: privacy-gateway-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: privacy-gateway
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: privacy-gateway
      maxAllowed:
        cpu: 4
        memory: 8Gi
      minAllowed:
        cpu: 500m
        memory: 1Gi
      controlledResources: ["cpu", "memory"]
```

## 🔍 Troubleshooting

### Common Issues

#### Pod Startup Issues
```bash
# Check pod status
kubectl get pods -n privacy-ml

# Check pod events
kubectl describe pod <pod-name> -n privacy-ml

# Check logs
kubectl logs <pod-name> -n privacy-ml --previous

# Check resource constraints
kubectl top pods -n privacy-ml
```

#### Performance Issues
```bash
# Check resource utilization
kubectl top nodes
kubectl top pods -A

# Check HPA status
kubectl get hpa -n privacy-ml

# Check VPA recommendations
kubectl describe vpa privacy-gateway-vpa -n privacy-ml
```

#### Networking Issues
```bash
# Test service connectivity
kubectl run test-pod --rm -i --tty --image=busybox -- /bin/sh
nslookup privacy-gateway-service.privacy-ml.svc.cluster.local

# Check ingress
kubectl get ingress -n privacy-ml
kubectl describe ingress privacy-ml-ingress -n privacy-ml

# Check network policies
kubectl get networkpolicies -n privacy-ml
```

### Log Analysis
```bash
# Centralized logging with ELK stack
kubectl logs -l app=privacy-gateway -n privacy-ml | grep "ERROR"

# Filter compliance violations
kubectl logs -l app=compliance-service -n privacy-ml | grep "VIOLATION"

# Monitor privacy budget usage
kubectl logs -l app=privacy-gateway -n privacy-ml | grep "BUDGET"
```

## 📞 Support & Maintenance

### Regular Maintenance Tasks
```bash
# Update container images
kubectl set image deployment/privacy-gateway privacy-gateway=privacy-ml/gateway:v1.2.0 -n privacy-ml

# Scale services
kubectl scale deployment privacy-gateway --replicas=10 -n privacy-ml

# Backup database
kubectl exec -it privacy-database-0 -n privacy-ml -- pg_dump privacy_ml > backup.sql

# Check certificate expiration
kubectl get secret privacy-ml-tls -n privacy-ml -o yaml | grep -A1 "tls.crt" | tail -1 | base64 -d | openssl x509 -noout -dates
```

### Performance Tuning
```bash
# Analyze slow queries
kubectl logs -l app=privacy-database -n privacy-ml | grep "slow query"

# Monitor memory usage
kubectl exec -it <pod-name> -n privacy-ml -- cat /proc/meminfo

# Check disk usage
kubectl exec -it <pod-name> -n privacy-ml -- df -h
```

For additional support, consult the [TERRAGON_IMPLEMENTATION_SUMMARY.md](TERRAGON_IMPLEMENTATION_SUMMARY.md) for detailed technical specifications and architecture information.

---

**Note**: This deployment guide covers the comprehensive setup of the Privacy-Preserving ML Framework with all TERRAGON enhancements including advanced research capabilities, enterprise security, intelligent scaling, quality gates, and global-first features.