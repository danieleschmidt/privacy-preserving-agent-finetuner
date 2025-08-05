#!/bin/bash
# Production startup script for Privacy-Preserving Agent Finetuner
# Enhanced with comprehensive production features

set -euo pipefail

# Configuration
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8080}"
export WORKERS="${WORKERS:-4}"
export MAX_CONNECTIONS="${MAX_CONNECTIONS:-1000}"
export KEEP_ALIVE="${KEEP_ALIVE:-2}"
export WORKER_CLASS="${WORKER_CLASS:-uvicorn.workers.UvicornWorker}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
export ACCESS_LOG="${ACCESS_LOG:-true}"
export GRACEFUL_TIMEOUT="${GRACEFUL_TIMEOUT:-120}"
export TIMEOUT="${TIMEOUT:-60}"

# Privacy and Security Configuration
export PRIVACY_LEVEL="${PRIVACY_LEVEL:-high}"
export AUDIT_LOGGING="${AUDIT_LOGGING:-true}"
export SECURITY_MONITORING="${SECURITY_MONITORING:-true}"
export ENCRYPTION_AT_REST="${ENCRYPTION_AT_REST:-true}"
export TLS_ENABLED="${TLS_ENABLED:-false}"

# Performance Configuration
export PRELOAD_APP="${PRELOAD_APP:-true}"
export WORKER_CONNECTIONS="${WORKER_CONNECTIONS:-1000}"
export MAX_REQUESTS="${MAX_REQUESTS:-1000}"
export MAX_REQUESTS_JITTER="${MAX_REQUESTS_JITTER:-100}"

# Monitoring Configuration
export PROMETHEUS_METRICS="${PROMETHEUS_METRICS:-true}"
export HEALTH_CHECK_ENABLED="${HEALTH_CHECK_ENABLED:-true}"
export PERFORMANCE_PROFILING="${PERFORMANCE_PROFILING:-false}"

# Database Configuration
export DATABASE_POOL_SIZE="${DATABASE_POOL_SIZE:-20}"
export DATABASE_MAX_OVERFLOW="${DATABASE_MAX_OVERFLOW:-30}"
export DATABASE_POOL_TIMEOUT="${DATABASE_POOL_TIMEOUT:-30}"

# Cache Configuration
export REDIS_MAX_CONNECTIONS="${REDIS_MAX_CONNECTIONS:-100}"
export CACHE_TTL="${CACHE_TTL:-3600}"

# Logging Configuration
LOG_DIR="/var/log/privacy-finetuner"
AUDIT_LOG_FILE="${LOG_DIR}/audit.log"
ACCESS_LOG_FILE="${LOG_DIR}/access.log"
ERROR_LOG_FILE="${LOG_DIR}/error.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Signal handlers for graceful shutdown
shutdown_handler() {
    log_info "Received shutdown signal, gracefully shutting down..."
    
    # Stop the main process
    if [ ! -z "${MAIN_PID:-}" ]; then
        kill -TERM "$MAIN_PID" 2>/dev/null || true
        wait "$MAIN_PID" 2>/dev/null || true
    fi
    
    # Stop monitoring processes
    if [ ! -z "${MONITOR_PID:-}" ]; then
        kill -TERM "$MONITOR_PID" 2>/dev/null || true
    fi
    
    log_success "Shutdown complete"
    exit 0
}

# Set up signal handlers
trap shutdown_handler SIGTERM SIGINT SIGQUIT

# Pre-flight checks
preflight_checks() {
    log_info "Starting pre-flight checks..."
    
    # Check Python version
    python_version=$(python --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
    if [[ $(echo "$python_version >= 3.9" | bc -l) -eq 0 ]]; then
        log_error "Python 3.9+ required, found $python_version"
        exit 1
    fi
    log_success "Python version check passed: $python_version"
    
    # Check required environment variables
    required_vars=("API_HOST" "API_PORT")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    log_success "Environment variables check passed"
    
    # Check disk space
    available_space=$(df /app | awk 'NR==2 {print $4}')
    min_space=1048576  # 1GB in KB
    if [ "$available_space" -lt "$min_space" ]; then
        log_warn "Low disk space: ${available_space}KB available"
    fi
    
    # Check memory
    available_memory=$(free -m | awk 'NR==2{print $7}')
    min_memory=512  # 512MB
    if [ "$available_memory" -lt "$min_memory" ]; then
        log_warn "Low memory: ${available_memory}MB available"
    fi
    
    # Test application import
    if ! python -c "import privacy_finetuner.api.server" 2>/dev/null; then
        log_error "Failed to import privacy_finetuner.api.server"
        exit 1
    fi
    log_success "Application import test passed"
    
    log_success "All pre-flight checks passed"
}

# Initialize logging
init_logging() {
    log_info "Initializing logging..."
    
    # Create log directories
    mkdir -p "$LOG_DIR"
    
    # Set up log rotation if logrotate is available
    if command -v logrotate >/dev/null 2>&1; then
        cat > /tmp/privacy-finetuner-logrotate.conf << EOF
$LOG_DIR/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 privacy privacy
    postrotate
        kill -USR1 \$(cat /tmp/privacy-finetuner.pid) 2>/dev/null || true
    endscript
}
EOF
        log_info "Log rotation configured"
    fi
    
    log_success "Logging initialized"
}

# Initialize security
init_security() {
    log_info "Initializing security features..."
    
    # Set secure file permissions
    find /app -type f -name "*.py" -exec chmod 644 {} \;
    find /app -type d -exec chmod 755 {} \;
    
    # Initialize audit logging
    if [ "$AUDIT_LOGGING" = "true" ]; then
        touch "$AUDIT_LOG_FILE"
        chmod 600 "$AUDIT_LOG_FILE"
        log_info "Audit logging enabled"
    fi
    
    # Initialize security monitoring
    if [ "$SECURITY_MONITORING" = "true" ]; then
        log_info "Security monitoring enabled"
    fi
    
    # Set up TLS if enabled
    if [ "$TLS_ENABLED" = "true" ]; then
        if [ -f "/app/certs/server.crt" ] && [ -f "/app/certs/server.key" ]; then
            export SSL_CERTFILE="/app/certs/server.crt"
            export SSL_KEYFILE="/app/certs/server.key"
            log_info "TLS enabled with certificates"
        else
            log_warn "TLS enabled but certificates not found, falling back to HTTP"
            export TLS_ENABLED="false"
        fi
    fi
    
    log_success "Security initialization complete"
}

# Initialize performance monitoring
init_monitoring() {
    log_info "Initializing performance monitoring..."
    
    # Start resource monitoring in background
    if [ "$PERFORMANCE_PROFILING" = "true" ]; then
        (
            while true; do
                {
                    echo "timestamp: $(date -Iseconds)"
                    echo "cpu_usage: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
                    echo "memory_usage: $(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}')"
                    echo "disk_usage: $(df /app | awk 'NR==2{print $5}')"
                    echo "---"
                } >> "$LOG_DIR/performance.log"
                sleep 60
            done
        ) &
        MONITOR_PID=$!
        log_info "Performance monitoring started (PID: $MONITOR_PID)"
    fi
    
    # Initialize Prometheus metrics if enabled
    if [ "$PROMETHEUS_METRICS" = "true" ]; then
        export ENABLE_PROMETHEUS="true"
        export PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
        log_info "Prometheus metrics enabled on port $PROMETHEUS_PORT"
    fi
    
    log_success "Monitoring initialization complete"
}

# Health check function
health_check() {
    if [ "$HEALTH_CHECK_ENABLED" = "true" ]; then
        local max_attempts=30
        local attempt=1
        
        log_info "Waiting for application to be ready..."
        
        while [ $attempt -le $max_attempts ]; do
            if curl -f -s "http://localhost:$API_PORT/health" >/dev/null 2>&1; then
                log_success "Application is ready and healthy"
                return 0
            fi
            
            log_info "Health check attempt $attempt/$max_attempts failed, retrying in 2 seconds..."
            sleep 2
            ((attempt++))
        done
        
        log_error "Application failed to become healthy after $max_attempts attempts"
        return 1
    fi
}

# Build Gunicorn command
build_gunicorn_cmd() {
    local cmd="gunicorn"
    
    # Basic configuration
    cmd="$cmd --bind $API_HOST:$API_PORT"
    cmd="$cmd --workers $WORKERS"
    cmd="$cmd --worker-class $WORKER_CLASS"
    cmd="$cmd --worker-connections $WORKER_CONNECTIONS"
    cmd="$cmd --max-requests $MAX_REQUESTS"
    cmd="$cmd --max-requests-jitter $MAX_REQUESTS_JITTER"
    cmd="$cmd --timeout $TIMEOUT"
    cmd="$cmd --graceful-timeout $GRACEFUL_TIMEOUT"
    cmd="$cmd --keep-alive $KEEP_ALIVE"
    
    # Preload app for better performance
    if [ "$PRELOAD_APP" = "true" ]; then
        cmd="$cmd --preload"
    fi
    
    # Logging
    cmd="$cmd --log-level $LOG_LEVEL"
    if [ "$ACCESS_LOG" = "true" ]; then
        cmd="$cmd --access-logfile $ACCESS_LOG_FILE"
    fi
    cmd="$cmd --error-logfile $ERROR_LOG_FILE"
    
    # PID file
    cmd="$cmd --pid /tmp/privacy-finetuner.pid"
    
    # TLS configuration
    if [ "$TLS_ENABLED" = "true" ]; then
        cmd="$cmd --certfile $SSL_CERTFILE --keyfile $SSL_KEYFILE"
    fi
    
    # Application module
    cmd="$cmd privacy_finetuner.api.server:app"
    
    echo "$cmd"
}

# Main startup function
main() {
    log_info "Starting Privacy-Preserving Agent Finetuner in production mode..."
    log_info "Version: $(python -c "import privacy_finetuner; print(privacy_finetuner.__version__)" 2>/dev/null || echo 'unknown')"
    log_info "Python: $(python --version)"
    log_info "Workers: $WORKERS"
    log_info "Host: $API_HOST:$API_PORT"
    log_info "Privacy Level: $PRIVACY_LEVEL"
    
    # Run initialization steps
    preflight_checks
    init_logging
    init_security
    init_monitoring
    
    # Build and execute Gunicorn command
    gunicorn_cmd=$(build_gunicorn_cmd)
    log_info "Starting server with command: $gunicorn_cmd"
    
    # Start the application
    exec $gunicorn_cmd &
    MAIN_PID=$!
    
    # Wait for the application to be ready
    if ! health_check; then
        log_error "Application startup failed"
        exit 1
    fi
    
    log_success "Privacy-Preserving Agent Finetuner started successfully!"
    log_info "Server is listening on $API_HOST:$API_PORT"
    log_info "Process ID: $MAIN_PID"
    
    # Show configuration summary
    echo
    echo "=== Configuration Summary ==="
    echo "API Host: $API_HOST"
    echo "API Port: $API_PORT"
    echo "Workers: $WORKERS"
    echo "Privacy Level: $PRIVACY_LEVEL"
    echo "TLS Enabled: $TLS_ENABLED"
    echo "Audit Logging: $AUDIT_LOGGING"
    echo "Security Monitoring: $SECURITY_MONITORING"
    echo "Prometheus Metrics: $PROMETHEUS_METRICS"
    echo "Performance Profiling: $PERFORMANCE_PROFILING"
    echo "Log Directory: $LOG_DIR"
    echo "=============================="
    echo
    
    # Wait for the main process
    wait $MAIN_PID
}

# Run main function
main "$@"