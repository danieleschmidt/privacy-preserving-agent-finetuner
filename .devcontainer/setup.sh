#!/bin/bash

# DevContainer setup script for Privacy-Preserving Agent Fine-Tuner
# This script runs after the container is created to set up the development environment

set -e

echo "🚀 Setting up Privacy-Preserving Agent Fine-Tuner development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    htop \
    tree \
    jq \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Poetry if not already installed
if ! command -v poetry &> /dev/null; then
    echo "📝 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="/home/vscode/.local/bin:$PATH"
    echo 'export PATH="/home/vscode/.local/bin:$PATH"' >> ~/.bashrc
    echo 'export PATH="/home/vscode/.local/bin:$PATH"' >> ~/.zshrc
fi

# Configure Poetry
echo "⚙️ Configuring Poetry..."
poetry config virtualenvs.create false
poetry config virtualenvs.in-project false

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
cd /workspace
poetry install --with dev,test

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Set up Jupyter kernel
echo "📚 Setting up Jupyter kernel..."
python -m ipykernel install --user --name privacy-finetuner --display-name "Privacy Fine-tuner"

# Create development directories
echo "📁 Creating development directories..."
mkdir -p /workspace/{logs,tmp,notebooks,experiments}
mkdir -p /workspace/data/{raw,processed,models,checkpoints}

# Set up environment file
echo "🔐 Setting up environment configuration..."
if [ ! -f /workspace/.env ]; then
    cp /workspace/.env.example /workspace/.env
    echo "Created .env file from template"
fi

# Initialize database (if PostgreSQL is available)
echo "🗄️ Checking database setup..."
if command -v psql &> /dev/null; then
    echo "PostgreSQL available - database setup can be run manually"
fi

# Set up monitoring directories
echo "📊 Setting up monitoring..."
mkdir -p /workspace/monitoring/logs
mkdir -p /workspace/monitoring/dashboards

# Configure Git
echo "🔧 Configuring Git..."
git config --global --add safe.directory /workspace
git config --global core.editor "code --wait"

# Install additional development tools
echo "🛠️ Installing additional development tools..."

# Install ruff (modern Python linter)
pip install --upgrade ruff

# Install mypy for type checking
pip install --upgrade mypy

# Install privacy-specific tools
pip install --upgrade \
    opacus \
    differential-privacy \
    tensorflow-privacy

# Set up shell aliases
echo "🐚 Setting up shell aliases..."
cat >> ~/.bashrc << 'ALIASES'

# Privacy Fine-tuner aliases
alias pf='python -m privacy_finetuner'
alias pf-train='python -m privacy_finetuner.cli train'
alias pf-eval='python -m privacy_finetuner.cli evaluate'
alias pf-server='python -m privacy_finetuner.api.server'
alias pf-test='pytest tests/ -v'
alias pf-lint='ruff check .'
alias pf-format='ruff format .'
alias pf-type='mypy privacy_finetuner/'
alias pf-security='bandit -r privacy_finetuner/'
alias pf-privacy='python scripts/privacy_compliance_check.py'

# Development shortcuts
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'

ALIASES

# Copy aliases to zsh if it exists
if [ -f ~/.zshrc ]; then
    cat ~/.bashrc | tail -n 20 >> ~/.zshrc
fi

# Set permissions
echo "🔐 Setting file permissions..."
chmod +x /workspace/scripts/*.py
chmod +x /workspace/.devcontainer/setup.sh

# Create startup message
cat > ~/.startup_message << 'MSG'
🔒 Privacy-Preserving Agent Fine-Tuner Development Environment

Quick Commands:
  pf-train      - Start privacy-preserving training
  pf-eval       - Evaluate model with privacy metrics
  pf-server     - Start API server
  pf-test       - Run test suite
  pf-lint       - Check code quality
  pf-privacy    - Run privacy compliance checks

Documentation: /workspace/docs/
Examples: /workspace/notebooks/
Logs: /workspace/logs/

Happy privacy-preserving ML development! 🚀
MSG

# Add startup message to shell profiles
echo 'cat ~/.startup_message' >> ~/.bashrc
echo 'cat ~/.startup_message' >> ~/.zshrc

# Final setup
echo "🎯 Finalizing setup..."

# Warm up the Python environment
python -c "import privacy_finetuner; print('✅ Privacy Fine-tuner package imported successfully')"

# Check CUDA availability
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('ℹ️  CUDA not available - using CPU mode')
"

# Check privacy libraries
python -c "
try:
    import opacus
    print('✅ Opacus (Differential Privacy) available')
except ImportError:
    print('❌ Opacus not available')

try:
    import transformers
    print('✅ Transformers available')
except ImportError:
    print('❌ Transformers not available')
"

echo ""
echo "✅ Development environment setup complete!"
echo "🔒 Privacy-Preserving Agent Fine-Tuner is ready for development"
echo ""
echo "Next steps:"
echo "  1. Review the configuration in .env"
echo "  2. Check out the examples in /workspace/notebooks/"
echo "  3. Run 'pf-test' to ensure everything works"
echo "  4. Start developing with privacy-first principles!"
echo ""