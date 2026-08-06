# Lumina - Educational Transformer Language Model

A comprehensive educational implementation of a transformer-based language model from scratch.

## Features

- **Tokenizer**: Character and BPE tokenization
- **Model**: Transformer architecture with multi-head attention
- **Training**: Full training pipeline with validation
- **Inference**: Text generation with multiple sampling strategies
- **API**: FastAPI backend for serving the model
- **CLI**: Command-line interface for training, generation, and chat

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python -m main train --config config.yaml --device cuda
```

### Text Generation

```bash
python -m main generate --checkpoint checkpoint.pt --prompt "Hello"
```

### Interactive Chat

```bash
python -m main chat --checkpoint checkpoint.pt
```

### API Server

```bash
python -m main api --checkpoint checkpoint.pt
```

Then visit `http://localhost:8000/docs` for the interactive API documentation.

## Project Structure

```
Lumina/
├── tokenizer/          # Tokenization modules
├── model/              # Model architecture
├── data/               # Data loading and preprocessing
├── training/           # Training utilities
├── inference/          # Inference pipeline
├── backend/            # Training and API backends
├── utils/              # Utility functions
├── config.yaml         # Configuration file
└── main.py            # Entry point
```

## Configuration

Edit `config.yaml` to customize:
- Model hyperparameters (embedding dim, layers, heads)
- Training settings (batch size, learning rate, epochs)
- Dataset path and tokenizer type
- Checkpoint directory

## Documentation

Each module contains detailed docstrings and type hints. Refer to individual files for more information.

## License

Educational use only.
