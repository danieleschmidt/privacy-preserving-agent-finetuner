# Multi-stage build for privacy-preserving agent finetuner
FROM python:3.11-slim as base

# Security: Run as non-root user
RUN groupadd -r privacy && useradd -r -g privacy privacy

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Poetry
RUN pip install poetry==1.7.1
ENV POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Development stage
FROM base as development
RUN poetry install --with dev,docs && rm -rf $POETRY_CACHE_DIR
COPY . .
USER privacy
EXPOSE 8080
CMD ["poetry", "run", "python", "-m", "privacy_finetuner.api.server"]

# Production dependencies stage
FROM base as deps
RUN poetry install --only=main --no-dev && rm -rf $POETRY_CACHE_DIR

# Security scanning stage
FROM deps as security-scan
COPY . .
RUN poetry run bandit -r privacy_finetuner/ -f json -o /tmp/bandit-report.json || true
RUN poetry run safety check --json --output /tmp/safety-report.json || true

# Production stage
FROM python:3.11-slim as production

# Security: Run as non-root user
RUN groupadd -r privacy && useradd -r -g privacy -s /bin/false privacy

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from deps stage
COPY --from=deps /app/.venv /app/.venv

# Ensure the virtual environment is in PATH
ENV PATH="/app/.venv/bin:$PATH"

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models \
    && chown -R privacy:privacy /app

WORKDIR /app

# Copy application code
COPY --chown=privacy:privacy . .

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Security: Switch to non-root user
USER privacy

# Expose port
EXPOSE 8080

# Set labels for metadata
LABEL maintainer="Daniel Schmidt <daniel@terragon-labs.com>" \
      version="0.1.0" \
      description="Privacy-Preserving Agent Finetuner" \
      vendor="Terragon Labs" \
      org.opencontainers.image.title="Privacy-Preserving Agent Finetuner" \
      org.opencontainers.image.description="Enterprise-grade framework for fine-tuning LLMs with differential privacy" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.url="https://github.com/terragon-labs/privacy-preserving-agent-finetuner" \
      org.opencontainers.image.source="https://github.com/terragon-labs/privacy-preserving-agent-finetuner" \
      org.opencontainers.image.licenses="MIT"

# Default command
CMD ["python", "-m", "privacy_finetuner.api.server"]