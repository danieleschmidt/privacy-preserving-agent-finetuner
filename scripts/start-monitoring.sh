#!/bin/bash

# Start monitoring stack for Privacy-Preserving Agent Fine-Tuner
# Comprehensive observability setup with privacy-specific metrics

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
COMPOSE_FILE="monitoring/docker-compose.monitoring.yml"
ENVIRONMENT="development"
DETACHED=true
SCALE_SERVICES=""

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --env ENV          Environment (development, staging, production) [default: development]"
    echo "  -f, --file FILE        Docker compose file [default: monitoring/docker-compose.monitoring.yml]"
    echo "  --foreground           Run in foreground (not detached)"
    echo "  --scale SERVICE=N      Scale specific services (e.g., --scale prometheus=2)"
    echo "  --stop                 Stop monitoring services"
    echo "  --restart              Restart monitoring services"
    echo "  --logs SERVICE         View logs for specific service"
    echo "  --status               Show status of monitoring services"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Start monitoring stack"
    echo "  $0 --env production --foreground     # Start in production mode, foreground"
    echo "  $0 --stop                           # Stop all monitoring services"
    echo "  $0 --logs grafana                   # View Grafana logs"
    echo "  $0 --scale prometheus=2             # Scale Prometheus to 2 instances"
}

# Parse command line arguments
ACTION="start"
LOG_SERVICE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -f|--file)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        --foreground)
            DETACHED=false
            shift
            ;;
        --scale)
            SCALE_SERVICES="$SCALE_SERVICES --scale $2"
            shift 2
            ;;
        --stop)
            ACTION="stop"
            shift
            ;;
        --restart)
            ACTION="restart"
            shift
            ;;
        --logs)
            ACTION="logs"
            LOG_SERVICE="$2"
            shift 2
            ;;
        --status)
            ACTION="status"
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

# Check if Docker and Docker Compose are available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not available${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}Error: Compose file not found: $COMPOSE_FILE${NC}"
    echo "Make sure you're running from the repository root."
    exit 1
fi

# Set environment variables based on environment
case $ENVIRONMENT in
    development)
        export GRAFANA_ADMIN_PASSWORD="admin_dev"
        export INFLUXDB_ADMIN_PASSWORD="admin_dev"
        export INFLUXDB_ADMIN_TOKEN="dev_token_change_in_production"
        ;;
    staging)
        export GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin_staging}"
        export INFLUXDB_ADMIN_PASSWORD="${INFLUXDB_ADMIN_PASSWORD:-admin_staging}"
        export INFLUXDB_ADMIN_TOKEN="${INFLUXDB_ADMIN_TOKEN:-staging_token}"
        ;;
    production)
        if [ -z "$GRAFANA_ADMIN_PASSWORD" ] || [ -z "$INFLUXDB_ADMIN_PASSWORD" ]; then
            echo -e "${RED}Error: Production environment requires GRAFANA_ADMIN_PASSWORD and INFLUXDB_ADMIN_PASSWORD${NC}"
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}Error: Invalid environment '$ENVIRONMENT'. Must be one of: development, staging, production${NC}"
        exit 1
        ;;
esac

# Execute action
case $ACTION in
    start)
        echo -e "${BLUE}🚀 Starting Privacy-Preserving ML Monitoring Stack${NC}"
        echo -e "${BLUE}Environment: $ENVIRONMENT${NC}"
        echo -e "${BLUE}Compose file: $COMPOSE_FILE${NC}"
        
        # Create necessary directories
        mkdir -p monitoring/grafana/{dashboards,datasources}
        mkdir -p monitoring/logs
        
        # Start services
        if [[ $DETACHED == true ]]; then
            docker compose -f "$COMPOSE_FILE" up -d $SCALE_SERVICES
        else
            docker compose -f "$COMPOSE_FILE" up $SCALE_SERVICES
        fi
        
        if [[ $DETACHED == true ]]; then
            echo -e "${GREEN}✅ Monitoring stack started successfully${NC}"
            echo ""
            echo -e "${BLUE}📊 Available Services:${NC}"
            echo "  Grafana:       http://localhost:3000 (admin/admin_dev)"
            echo "  Prometheus:    http://localhost:9090"
            echo "  AlertManager:  http://localhost:9093"
            echo "  Loki:          http://localhost:3100"
            echo "  Jaeger:        http://localhost:16686"
            echo "  Kibana:        http://localhost:5601"
            echo "  InfluxDB:      http://localhost:8086"
            echo ""
            echo -e "${YELLOW}⚠️  Change default passwords in production!${NC}"
            echo ""
            echo -e "${BLUE}Next steps:${NC}"
            echo "  1. Import Grafana dashboards from monitoring/grafana/dashboards/"
            echo "  2. Configure alert notification channels"
            echo "  3. Set up privacy budget thresholds"
            echo "  4. Test privacy violation alerts"
        fi
        ;;
        
    stop)
        echo -e "${YELLOW}🛑 Stopping monitoring services...${NC}"
        docker compose -f "$COMPOSE_FILE" down
        echo -e "${GREEN}✅ Monitoring stack stopped${NC}"
        ;;
        
    restart)
        echo -e "${YELLOW}🔄 Restarting monitoring services...${NC}"
        docker compose -f "$COMPOSE_FILE" restart
        echo -e "${GREEN}✅ Monitoring stack restarted${NC}"
        ;;
        
    logs)
        if [ -z "$LOG_SERVICE" ]; then
            echo -e "${BLUE}📋 Showing logs for all services:${NC}"
            docker compose -f "$COMPOSE_FILE" logs -f
        else
            echo -e "${BLUE}📋 Showing logs for service: $LOG_SERVICE${NC}"
            docker compose -f "$COMPOSE_FILE" logs -f "$LOG_SERVICE"
        fi
        ;;
        
    status)
        echo -e "${BLUE}📊 Monitoring Stack Status:${NC}"
        docker compose -f "$COMPOSE_FILE" ps
        echo ""
        
        # Check health of key services
        echo -e "${BLUE}🏥 Health Checks:${NC}"
        
        # Prometheus
        if curl -s http://localhost:9090/-/healthy &> /dev/null; then
            echo -e "  Prometheus:    ${GREEN}✅ Healthy${NC}"
        else
            echo -e "  Prometheus:    ${RED}❌ Unhealthy${NC}"
        fi
        
        # Grafana
        if curl -s http://localhost:3000/api/health &> /dev/null; then
            echo -e "  Grafana:       ${GREEN}✅ Healthy${NC}"
        else
            echo -e "  Grafana:       ${RED}❌ Unhealthy${NC}"
        fi
        
        # ElasticSearch
        if curl -s http://localhost:9200/_cluster/health &> /dev/null; then
            echo -e "  ElasticSearch: ${GREEN}✅ Healthy${NC}"
        else
            echo -e "  ElasticSearch: ${RED}❌ Unhealthy${NC}"
        fi
        
        echo ""
        echo -e "${BLUE}💾 Volume Usage:${NC}"
        docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}\t{{.Reclaimable}}"
        ;;
esac

# Display tips for different environments
if [[ $ACTION == "start" && $DETACHED == true ]]; then
    echo -e "${BLUE}💡 Tips for $ENVIRONMENT environment:${NC}"
    
    case $ENVIRONMENT in
        development)
            echo "  - Use default credentials for quick access"
            echo "  - All data is ephemeral - use volumes for persistence"
            echo "  - Privacy alerts are configured for development thresholds"
            ;;
        staging)
            echo "  - Test production alert configurations"
            echo "  - Validate privacy budget tracking accuracy"
            echo "  - Verify compliance monitoring works correctly"
            ;;
        production)
            echo "  - Ensure all passwords are secure and rotated"
            echo "  - Configure external storage for data persistence"
            echo "  - Set up backup and disaster recovery procedures"
            echo "  - Monitor resource usage and scale as needed"
            ;;
    esac
    
    echo ""
    echo -e "${BLUE}🔧 Maintenance commands:${NC}"
    echo "  $0 --status              # Check service health"
    echo "  $0 --logs grafana        # View specific service logs"
    echo "  $0 --restart             # Restart all services"
    echo "  $0 --stop                # Stop all services"
fi