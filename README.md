# Lumina – An Educational Transformer LLM from Scratch

A complete, well-documented educational transformer-based language model built entirely in Python. Lumina demonstrates how small language models work—from tokenization and dataset handling to training, checkpointing, and text generation.

## 🎯 Project Goals

- **Educational**: Clear, modular code with extensive comments and documentation
- **Self-contained**: No external model downloads—train from your own .txt datasets
- **Interactive**: Modern web UI for training, monitoring, and inference
- **Extensible**: Easy to modify, experiment with, and extend
- **Complete**: Full pipeline from raw text to trained model

## ✨ Features

### Core Components
- **Transformer Decoder Model**: Multi-head self-attention, feed-forward layers, residual connections
- **Tokenizer**: Vocabulary builder with UTF-8 support
- **Training Engine**: Full training loop with validation, checkpointing, and live progress
- **Dataset System**: Automatic .txt file detection, merging, and statistics
- **Inference Pipeline**: Temperature, top-k, and top-p sampling for text generation

### User Interface
- **Chat Interface**: Interact with trained models
- **Training Dashboard**: Real-time progress, loss graphs, and statistics
- **Dataset Manager**: Browse, add, remove, and preview datasets
- **Model Management**: Save, load, and export checkpoints
- **Configuration Panel**: Adjust hyperparameters

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip or conda
- ~4GB RAM (minimum)
- ~2GB disk space for datasets and checkpoints

### Installation

```bash
# Clone the repository
git clone https://github.com/editcoustomer-max/Lumina.git
cd Lumina

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Lumina

#### Terminal-First Approach (Recommended for Learning)

```bash
# 1. Prepare your datasets
# Place .txt files in datasets/ folder (see Dataset Structure below)

# 2. Train the model (terminal interface)
python run.py --mode train --config config/training.yaml

# 3. Generate text
python run.py --mode generate --prompt "Hello" --config config/inference.yaml

# 4. Interactive chat
python run.py --mode chat --config config/inference.yaml
```

#### Web Interface

```bash
# Start the backend API
python backend/api.py

# In another terminal, start the frontend
python -m streamlit run frontend/app.py
```

## 📁 Project Structure

```
Lumina/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── run.py                         # Main entry point
│
├── backend/
│   ├── __init__.py
│   ├── api.py                    # FastAPI backend
│   ├── trainer.py                # Training orchestration
│   ├── inference_engine.py        # Text generation engine
│   └── utils.py                  # Backend utilities
│
├── frontend/
│   ├── __init__.py
│   ├── app.py                    # Streamlit UI
│   ├── pages/
│   │   ├── chat.py
│   │   ├── training.py
│   │   ├── dataset_manager.py
│   │   ├── model_stats.py
│   │   └── settings.py
│   └── components/
│       ├── dataset_viewer.py
│       ├── training_monitor.py
│       └── loss_graph.py
│
├── model/
│   ├── __init__.py
│   ├── transformer.py            # Transformer architecture
│   ├── layers.py                 # Attention, feed-forward, etc.
│   └── checkpoint.py             # Model saving/loading
│
├── tokenizer/
│   ├── __init__.py
│   ├── tokenizer.py              # Tokenizer implementation
│   ├── vocab_builder.py          # Vocabulary management
│   └── embeddings.py             # Embedding layers
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                # Training loop
│   ├── validator.py              # Validation logic
│   ├── scheduler.py              # Learning rate scheduling
│   └── loss.py                   # Loss functions
│
├── inference/
│   ├── __init__.py
│   ├── sampler.py                # Sampling strategies
│   ├── generator.py              # Text generation
│   └── pipeline.py               # End-to-end inference
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                # Dataset loading
│   ├── loader.py                 # Data loaders
│   └── preprocessor.py           # Text preprocessing
│
├── datasets/                      # User datasets
│   ├── greetings/
│   ├── chat/
│   ├── books/
│   ├── encyclopedia/
│   ├── science/
│   ├── history/
│   ├── mathematics/
│   ├── programming/
│   ├── stories/
│   ├── conversations/
│   ├── question_answer/
│   └── custom/
│
├── checkpoints/                   # Saved models and tokenizers
│   └── .gitkeep
│
├── config/
│   ├── model.yaml                # Model config
│   ├── training.yaml             # Training config
│   ├── inference.yaml            # Inference config
│   └── default.yaml              # Default config
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                 # Logging utilities
│   ├── device.py                 # GPU/CPU detection
│   ├── metrics.py                # Training metrics
│   └── helpers.py                # General helpers
│
├── docs/
│   ├── ARCHITECTURE.md           # Architecture overview
│   ├── DATASET_GUIDE.md          # Dataset preparation
│   ├── TRAINING_GUIDE.md         # Training instructions
│   ├── INFERENCE_GUIDE.md        # Inference guide
│   └── TROUBLESHOOTING.md        # Common issues
│
└── .gitignore                    # Git ignore rules
```

## 📊 Dataset Structure

Add your .txt files to the `datasets/` folder. Lumina will automatically scan and detect them:

```
datasets/
├── greetings/
│   ├── hello.txt
│   └── welcome.txt
├── chat/
│   └── conversations.txt
├── books/
│   └── sample_book.txt
├── programming/
│   ├── python_tips.txt
│   └── javascript_guide.txt
└── custom/
    └── your_data.txt
```

**Dataset Requirements:**
- Files must be `.txt` format
- UTF-8 encoding
- Text content only (no binary data)
- One file can have any size (will be split into batches)
- Subdirectories are automatically organized by category

## 🎓 Training a Model

### Step 1: Add Datasets
Place .txt files in the `datasets/` folder. You can organize them by category.

### Step 2: Configure Training (Optional)
Edit `config/training.yaml` to adjust:
- Model size (hidden_dim, num_heads, num_layers)
- Learning rate and scheduler
- Batch size and sequence length
- Number of epochs
- Checkpoint frequency

### Step 3: Start Training

**Terminal:**
```bash
python run.py --mode train --config config/training.yaml
```

**Web UI:**
1. Open the Training page
2. Click "Load Dataset"
3. Review dataset statistics
4. Click "Start Training"
5. Monitor progress in real-time

### Step 4: Save Checkpoint
Checkpoints are saved automatically during training. Access them in:
- `checkpoints/` folder (terminal)
- Model Management page (web UI)

## 🗣️ Generating Text

### Via Terminal

```bash
# Generate completion for a prompt
python run.py --mode generate --prompt "Once upon a time" --length 100

# Interactive chat
python run.py --mode chat
```

### Via Web UI

1. Open the Chat page
2. Select a checkpoint
3. Type a prompt
4. Adjust temperature, top-k, top-p if desired
5. Click Generate

## ⚙️ Configuration

All configurations are in YAML format in the `config/` folder:

### `model.yaml`
```yaml
model:
  embedding_dim: 256
  hidden_dim: 512
  num_heads: 8
  num_layers: 4
  ff_dim: 2048
  dropout: 0.1
  vocab_size: 10000
  max_seq_length: 512
```

### `training.yaml`
```yaml
training:
  batch_size: 32
  learning_rate: 0.0005
  epochs: 10
  warmup_steps: 1000
  gradient_clip: 1.0
  checkpoint_every: 500
```

### `inference.yaml`
```yaml
inference:
  temperature: 0.7
  top_k: 50
  top_p: 0.9
  max_length: 256
```

## 📈 Monitoring Training

The training dashboard shows:
- **Loss Curve**: Training and validation loss over time
- **Learning Rate**: Current and scheduled learning rate
- **Speed**: Tokens per second, estimated time remaining
- **Dataset Stats**: Characters, words, tokens
- **Model Checkpoints**: Saved models with performance metrics

## 🔧 Architecture Overview

### Transformer Model
```
Input Text
    ↓
Tokenizer
    ↓
Token IDs → Embedding + Positional Encoding
    ↓
Transformer Block (repeat N times):
  ├─ Multi-Head Self-Attention
  ├─ Layer Norm + Residual
  ├─ Feed-Forward Network
  └─ Layer Norm + Residual
    ↓
Output Layer (vocab prediction)
    ↓
Softmax → Token Probabilities
    ↓
Sampler (temperature, top-k, top-p)
    ↓
Generated Text
```

### Training Loop
```
Load Dataset
    ↓
Initialize Model & Tokenizer
    ↓
For each epoch:
    ├─ Shuffle data
    ├─ For each batch:
    │  ├─ Forward pass
    │  ├─ Compute loss
    │  ├─ Backward pass
    │  ├─ Update weights
    │  └─ Log metrics
    ├─ Validate on held-out set
    └─ Save checkpoint
```

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Detailed model and system architecture
- **[DATASET_GUIDE.md](docs/DATASET_GUIDE.md)**: How to prepare and format datasets
- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)**: Step-by-step training instructions
- **[INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md)**: Text generation and sampling
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**: Common issues and solutions

## 🛠️ Requirements

See `requirements.txt` for full list. Main dependencies:

```
Python 3.10+
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0
rich>=13.0.0
streamlit>=1.28.0
fastapi>=0.100.0
uvicorn>=0.23.0
matplotlib>=3.7.0
```

## 💻 System Requirements

### Minimum
- CPU: Dual-core processor
- RAM: 4GB
- Storage: 2GB (for datasets + checkpoints)
- Python 3.10+

### Recommended
- CPU: Quad-core or better
- GPU: NVIDIA CUDA 11.8+ (optional, but ~10x faster)
- RAM: 8GB+
- Storage: 10GB+
- Python 3.11 or 3.12

## 🚀 Performance Tips

1. **Use GPU**: Install PyTorch with CUDA support for ~10x speedup
2. **Batch Size**: Increase for better hardware utilization (if memory allows)
3. **Sequence Length**: Shorter sequences = faster training, longer = better context
4. **Model Size**: Start small (2-4 layers), scale up as needed
5. **Dataset Size**: Quality > quantity. Well-curated text trains better than large raw dumps

## 📖 Learning Path

1. **Understand the Architecture** → Read `docs/ARCHITECTURE.md`
2. **Prepare Your Dataset** → Follow `docs/DATASET_GUIDE.md`
3. **Train a Small Model** → Use `docs/TRAINING_GUIDE.md`
4. **Generate Text** → Explore `docs/INFERENCE_GUIDE.md`
5. **Experiment** → Modify configs, try different datasets, adjust hyperparameters

## 🤝 Contributing

This is an educational project. Feel free to:
- Modify the code for learning purposes
- Extend functionality
- Experiment with different architectures
- Share improvements and fixes

## ⚠️ Limitations & Notes

- **Educational Only**: Not designed for production use or commercial purposes
- **Dataset Size**: Works best with 1MB-1GB of text (larger datasets benefit from more training time)
- **Model Scale**: Designed for learning, not state-of-the-art performance
- **Hardware**: CPU training is slow but works; GPU recommended
- **Memory**: Large models require more RAM; adjust config if running out of memory

## 📄 License

This project is provided as-is for educational purposes.

## 🙋 Support & Questions

For issues, questions, or suggestions:
1. Check `docs/TROUBLESHOOTING.md`
2. Review existing code comments
3. Experiment with the configuration files
4. Read the documentation in `docs/`

## 🎉 Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Add your first dataset: Place .txt files in `datasets/custom/`
3. Train: `python run.py --mode train`
4. Generate: `python run.py --mode generate --prompt "Hello"`
5. Explore: Open the web UI with `python -m streamlit run frontend/app.py`

Happy learning! 🚀
