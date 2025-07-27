.PHONY: help install dev-install test lint format security clean build docs serve-docs deploy-docs docker-build docker-run privacy-check compliance-check

# Variables
PYTHON := python3
POETRY := poetry
DOCKER := docker
IMAGE_NAME := privacy-finetuner
VERSION := $(shell $(POETRY) version -s)
REGISTRY := ghcr.io/terragon-labs

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(CYAN)Privacy-Preserving Agent Finetuner - Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(POETRY) install --only main

dev-install: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(POETRY) install --with dev,docs
	$(POETRY) run pre-commit install
	@echo "$(GREEN)Development environment setup complete!$(NC)"

test: ## Run test suite
	@echo "$(BLUE)Running test suite...$(NC)"
	$(POETRY) run pytest tests/ -v --cov=privacy_finetuner --cov-report=term-missing --cov-report=html --cov-report=xml

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(POETRY) run pytest tests/unit/ -v -m "not slow"

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(POETRY) run pytest tests/integration/ -v

test-privacy: ## Run privacy guarantee tests
	@echo "$(BLUE)Running privacy guarantee tests...$(NC)"
	$(POETRY) run pytest tests/privacy/ -v --privacy-budget=1.0

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(POETRY) run pytest tests/performance/ -v -m "slow"

lint: ## Run all linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	$(POETRY) run ruff check .
	$(POETRY) run black --check .
	$(POETRY) run isort --check-only .
	$(POETRY) run mypy privacy_finetuner/
	$(POETRY) run flake8 privacy_finetuner/
	@echo "$(GREEN)All linting checks passed!$(NC)"

format: ## Format code using black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(POETRY) run black .
	$(POETRY) run isort .
	$(POETRY) run ruff --fix .
	@echo "$(GREEN)Code formatting complete!$(NC)"

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	$(POETRY) run bandit -r privacy_finetuner/ -f json -o bandit-report.json
	$(POETRY) run safety check --json --output safety-report.json
	$(PYTHON) scripts/security_audit.py
	@echo "$(GREEN)Security checks complete!$(NC)"

privacy-check: ## Run privacy compliance checks
	@echo "$(BLUE)Running privacy compliance checks...$(NC)"
	$(PYTHON) scripts/privacy_compliance_check.py
	@echo "$(GREEN)Privacy compliance checks complete!$(NC)"

compliance-check: ## Run full compliance audit
	@echo "$(BLUE)Running compliance audit...$(NC)"
	$(PYTHON) scripts/compliance_audit.py
	@echo "$(GREEN)Compliance audit complete!$(NC)"

clean: ## Clean build artifacts and cache
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Clean complete!$(NC)"

build: ## Build package
	@echo "$(BLUE)Building package...$(NC)"
	$(POETRY) build
	@echo "$(GREEN)Package built successfully!$(NC)"

publish: ## Publish package to PyPI
	@echo "$(BLUE)Publishing package...$(NC)"
	$(POETRY) publish
	@echo "$(GREEN)Package published successfully!$(NC)"

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	$(POETRY) run mkdocs build
	@echo "$(GREEN)Documentation built successfully!$(NC)"

serve-docs: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	$(POETRY) run mkdocs serve

deploy-docs: ## Deploy documentation to GitHub Pages
	@echo "$(BLUE)Deploying documentation...$(NC)"
	$(POETRY) run mkdocs gh-deploy --force
	@echo "$(GREEN)Documentation deployed successfully!$(NC)"

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	$(DOCKER) build -t $(IMAGE_NAME):$(VERSION) -t $(IMAGE_NAME):latest .
	@echo "$(GREEN)Docker image built successfully!$(NC)"

docker-build-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(NC)"
	$(DOCKER) build -f Dockerfile.dev -t $(IMAGE_NAME):dev .
	@echo "$(GREEN)Development Docker image built successfully!$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	$(DOCKER) run -it --rm -p 8080:8080 -v $(PWD):/app $(IMAGE_NAME):latest

docker-run-dev: ## Run development Docker container
	@echo "$(BLUE)Running development Docker container...$(NC)"
	$(DOCKER) run -it --rm -p 8080:8080 -v $(PWD):/app $(IMAGE_NAME):dev

docker-push: ## Push Docker image to registry
	@echo "$(BLUE)Pushing Docker image to registry...$(NC)"
	$(DOCKER) tag $(IMAGE_NAME):$(VERSION) $(REGISTRY)/$(IMAGE_NAME):$(VERSION)
	$(DOCKER) tag $(IMAGE_NAME):latest $(REGISTRY)/$(IMAGE_NAME):latest
	$(DOCKER) push $(REGISTRY)/$(IMAGE_NAME):$(VERSION)
	$(DOCKER) push $(REGISTRY)/$(IMAGE_NAME):latest
	@echo "$(GREEN)Docker image pushed successfully!$(NC)"

docker-compose-up: ## Start services with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Services started successfully!$(NC)"

docker-compose-down: ## Stop services with docker-compose
	@echo "$(BLUE)Stopping services with docker-compose...$(NC)"
	docker-compose down
	@echo "$(GREEN)Services stopped successfully!$(NC)"

setup-dev: dev-install ## Complete development setup
	@echo "$(BLUE)Setting up development environment...$(NC)"
	cp .env.example .env
	@echo "$(YELLOW)Please update .env file with your configuration$(NC)"
	@echo "$(GREEN)Development setup complete!$(NC)"

check: lint test security privacy-check ## Run all checks
	@echo "$(GREEN)All checks passed successfully!$(NC)"

ci: ## Run CI pipeline locally
	@echo "$(BLUE)Running CI pipeline...$(NC)"
	$(MAKE) check
	$(MAKE) build
	@echo "$(GREEN)CI pipeline completed successfully!$(NC)"

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	$(POETRY) run python scripts/benchmark.py
	@echo "$(GREEN)Benchmarks complete!$(NC)"

profile: ## Profile application performance
	@echo "$(BLUE)Profiling application...$(NC)"
	$(POETRY) run python -m cProfile -o profile.stats scripts/profile_app.py
	$(POETRY) run python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
	@echo "$(GREEN)Profiling complete!$(NC)"

monitor: ## Start monitoring stack
	@echo "$(BLUE)Starting monitoring stack...$(NC)"
	docker-compose -f docker-compose.monitoring.yml up -d
	@echo "$(GREEN)Monitoring stack started at:$(NC)"
	@echo "  $(CYAN)Prometheus: http://localhost:9090$(NC)"
	@echo "  $(CYAN)Grafana: http://localhost:3000$(NC)"
	@echo "  $(CYAN)Jaeger: http://localhost:16686$(NC)"

stop-monitor: ## Stop monitoring stack
	@echo "$(BLUE)Stopping monitoring stack...$(NC)"
	docker-compose -f docker-compose.monitoring.yml down
	@echo "$(GREEN)Monitoring stack stopped!$(NC)"

release: ## Create a new release
	@echo "$(BLUE)Creating new release...$(NC)"
	$(POETRY) run cz bump --changelog
	git push origin main --tags
	@echo "$(GREEN)Release created successfully!$(NC)"

backup: ## Backup important data
	@echo "$(BLUE)Creating backup...$(NC)"
	$(PYTHON) scripts/backup.py
	@echo "$(GREEN)Backup complete!$(NC)"

restore: ## Restore from backup
	@echo "$(BLUE)Restoring from backup...$(NC)"
	$(PYTHON) scripts/restore.py
	@echo "$(GREEN)Restore complete!$(NC)"

health-check: ## Check application health
	@echo "$(BLUE)Checking application health...$(NC)"
	curl -f http://localhost:8080/health || echo "$(RED)Health check failed!$(NC)"
	@echo "$(GREEN)Health check complete!$(NC)"

logs: ## Show application logs
	@echo "$(BLUE)Showing application logs...$(NC)"
	docker-compose logs -f privacy-finetuner

update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(POETRY) update
	$(POETRY) run pre-commit autoupdate
	@echo "$(GREEN)Dependencies updated!$(NC)"

audit: ## Run security and compliance audit
	@echo "$(BLUE)Running security and compliance audit...$(NC)"
	$(MAKE) security
	$(MAKE) compliance-check
	@echo "$(GREEN)Audit complete!$(NC)"

deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(NC)"
	$(PYTHON) scripts/deploy.py --env staging
	@echo "$(GREEN)Deployment to staging complete!$(NC)"

deploy-prod: ## Deploy to production environment
	@echo "$(BLUE)Deploying to production...$(NC)"
	$(PYTHON) scripts/deploy.py --env production
	@echo "$(GREEN)Deployment to production complete!$(NC)"

rollback: ## Rollback to previous version
	@echo "$(BLUE)Rolling back to previous version...$(NC)"
	$(PYTHON) scripts/rollback.py
	@echo "$(GREEN)Rollback complete!$(NC)"

status: ## Show application status
	@echo "$(BLUE)Application Status:$(NC)"
	@echo "$(CYAN)Version:$(NC) $(VERSION)"
	@echo "$(CYAN)Git Branch:$(NC) $$(git rev-parse --abbrev-ref HEAD)"
	@echo "$(CYAN)Git Commit:$(NC) $$(git rev-parse --short HEAD)"
	@echo "$(CYAN)Docker Images:$(NC)"
	@$(DOCKER) images | grep $(IMAGE_NAME) || echo "  No images found"
	@echo "$(CYAN)Running Containers:$(NC)"
	@$(DOCKER) ps | grep $(IMAGE_NAME) || echo "  No containers running"