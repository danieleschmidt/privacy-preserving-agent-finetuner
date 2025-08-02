# Installation Guide

This guide will help you install and set up the Privacy-Preserving Agent Fine-Tuner in various environments.

## Quick Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU support)
- Docker (for containerized deployment)
- Poetry (recommended) or pip

### Install from PyPI

```bash
pip install privacy-preserving-agent-finetuner
```

### Install from Source

```bash
git clone https://github.com/terragon-labs/privacy-preserving-agent-finetuner
cd privacy-preserving-agent-finetuner
poetry install
```

### Docker Installation

```bash
docker pull terragon-labs/privacy-finetuner:latest
docker run -it --gpus all terragon-labs/privacy-finetuner:latest
```

## Detailed Installation

See [SETUP_GUIDE.md](../../SETUP_GUIDE.md) for detailed installation instructions.

## Next Steps

After installation, see:
- [First Training Session](first-training.md) - Run your first privacy-preserving training
- [Basic Configuration](basic-config.md) - Configure essential settings