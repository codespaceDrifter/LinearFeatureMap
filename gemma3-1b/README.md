# Gemma 3 1B - Chat & Layer Access

Simple setup for running Google's Gemma 3 1B model in PyTorch.

## Quick Start

```bash
# 1. Setup
cd gemma3-1b
./setup.sh

# 2. Download weights (one time)
source venv/bin/activate
huggingface-cli download google/gemma-3-1b-pt --local-dir weights/

# 3. Chat!
python chat.py
```

## What's Here

### Main Files
- **`chat.py`** - Simple chat interface (just run this!)
- **`gemma_pytorch/`** - Official Google implementation with actual source code

### Key Source Files (Read the actual code!)
- `gemma_pytorch/gemma/model.py` - THE REAL MODEL (RMSNorm, Attention, MLP)
- `gemma_pytorch/gemma/config.py` - Model configuration
- `gemma_pytorch/gemma/tokenizer.py` - Tokenizer

## Architecture

- **26 layers** (5 local + 1 global attention, repeating)
- **Hidden size:** 1152
- **8 attention heads**, 2 KV heads (Grouped Query Attention)
- **Vocab:** 256K tokens
- **Max context:** 32K tokens

## For Layer-by-Layer Access

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./weights", output_hidden_states=True)

# Get all layer outputs
outputs = model(**inputs, output_hidden_states=True)
all_layers = outputs.hidden_states  # Tuple of (layer0, layer1, ..., layer26)
```

Or read `gemma_pytorch/gemma/model.py` to see how each layer works!

## Files Explained

- `setup.sh` - One-command setup
- `chat.py` - Main chat script
- `requirements.txt` - Python dependencies
- `gemma_pytorch/` - Official Google code
- `venv/` - Virtual environment (created by setup.sh)
- `weights/` - Model weights (you download these)
