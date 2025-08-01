#!/bin/bash

# Jupyter startup script for Privacy-Preserving ML development
set -e

echo "🔒 Starting Privacy-Preserving ML Jupyter Environment"

# Create necessary directories
mkdir -p /workspace/{notebooks,data,models,experiments,results,logs}

# Set up privacy research environment
echo "📊 Setting up privacy research environment..."

# Configure Git if not already configured
if [ ! -f ~/.gitconfig ]; then
    echo "⚙️ Configuring Git for development..."
    git config --global user.name "Privacy Researcher"
    git config --global user.email "researcher@privacy-ml.dev"
    git config --global init.defaultBranch main
fi

# Install additional research packages if needed
echo "📦 Checking for additional research packages..."

# Set up privacy examples and tutorials
if [ ! -f /workspace/notebooks/01_Getting_Started.ipynb ]; then
    echo "📚 Setting up example notebooks..."
    
    # Create getting started notebook
    cat > /workspace/notebooks/01_Getting_Started.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Privacy-Preserving Agent Fine-Tuner - Getting Started\n",
    "\n",
    "Welcome to the Privacy-Preserving Agent Fine-Tuner research environment!\n",
    "\n",
    "This notebook will guide you through the basics of privacy-preserving machine learning."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import the privacy finetuner library\n",
    "import privacy_finetuner\n",
    "from privacy_finetuner.core.privacy_config import PrivacyConfig\n",
    "from privacy_finetuner.core.context_guard import ContextGuard\n",
    "\n",
    "print(\"🔒 Privacy-Preserving Agent Fine-Tuner loaded successfully!\")\n",
    "print(f\"Version: {privacy_finetuner.__version__ if hasattr(privacy_finetuner, '__version__') else 'development'}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Privacy Configuration\n",
    "\n",
    "Let's start by configuring privacy parameters:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configure differential privacy parameters\n",
    "privacy_config = PrivacyConfig(\n",
    "    epsilon=1.0,        # Privacy budget\n",
    "    delta=1e-5,         # Privacy parameter\n",
    "    max_grad_norm=1.0,  # Gradient clipping threshold\n",
    "    noise_multiplier=0.5  # Noise scale for DP-SGD\n",
    ")\n",
    "\n",
    "print(f\"Privacy configuration: ε={privacy_config.epsilon}, δ={privacy_config.delta}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Context Protection\n",
    "\n",
    "Test the context protection capabilities:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize context guard for PII protection\n",
    "guard = ContextGuard(strategies=['pii_removal', 'entity_hashing'])\n",
    "\n",
    "# Test with sensitive text\n",
    "sensitive_text = \"John Doe with SSN 123-45-6789 requested access to his account.\"\n",
    "protected_text = guard.protect(sensitive_text, sensitivity_level=\"high\")\n",
    "\n",
    "print(f\"Original: {sensitive_text}\")\n",
    "print(f\"Protected: {protected_text}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Next Steps\n",
    "\n",
    "Explore the other example notebooks:\n",
    "\n",
    "- `02_Differential_Privacy_Training.ipynb` - Learn about DP-SGD training\n",
    "- `03_Federated_Learning.ipynb` - Federated learning with privacy\n",
    "- `04_Privacy_Analysis.ipynb` - Analyzing privacy-utility trade-offs\n",
    "- `05_Compliance_Validation.ipynb` - GDPR and HIPAA compliance\n",
    "\n",
    "Happy privacy-preserving ML research! 🚀"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Privacy-Preserving ML",
   "language": "python",
   "name": "privacy-ml"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
fi

# Set up model downloads directory
echo "🤖 Setting up model cache..."
export TRANSFORMERS_CACHE=/workspace/models/.cache
export HF_HOME=/workspace/models/.cache/huggingface
mkdir -p $TRANSFORMERS_CACHE $HF_HOME

# Set up privacy monitoring
echo "📊 Starting privacy monitoring..."
mkdir -p /workspace/logs/privacy

# Display startup information
echo ""
echo "🎯 Privacy-Preserving ML Research Environment Ready!"
echo ""
echo "📁 Workspace structure:"
echo "  /workspace/notebooks/    - Jupyter notebooks and tutorials"
echo "  /workspace/data/         - Training data and datasets"
echo "  /workspace/models/       - Model cache and checkpoints"
echo "  /workspace/experiments/  - Experiment results and logs"
echo "  /workspace/results/      - Research outputs and papers"
echo ""
echo "🔗 Available services:"
echo "  - Jupyter Lab: http://localhost:8888"
echo "  - TensorBoard: tensorboard --logdir /workspace/experiments/"
echo "  - W&B Dashboard: wandb login (if using Weights & Biases)"
echo ""
echo "🔒 Privacy Features:"
echo "  - Differential Privacy (Opacus)"
echo "  - Federated Learning (Flower)"
echo "  - Context Protection (PII removal)"
echo "  - Compliance Validation (GDPR/HIPAA)"
echo ""
echo "📚 Example notebooks available in /workspace/notebooks/"
echo ""

# Set up privacy environment variables
export PRIVACY_FINETUNER_ENV=jupyter
export PRIVACY_RESEARCH_MODE=true
export OPACUS_DISABLE_TENSOR_CHECKING=true  # For development speed

# Start Jupyter Lab
echo "🚀 Starting Jupyter Lab..."
exec jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --notebook-dir=/workspace \
    --ServerApp.token="${JUPYTER_TOKEN:-dev-token-change-in-production}" \
    --ServerApp.password='' \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.allow_origin='*' \
    --ServerApp.trust_xheaders=True