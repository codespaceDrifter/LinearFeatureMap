# Gemma 3 1B - Layer by Layer Inference

This directory contains everything you need to run Google's Gemma 3 1B model layer-by-layer in PyTorch with full access to activations.

## Setup

### 1. Activate the virtual environment

```bash
source venv/bin/activate
```

### 2. Install dependencies (if not already done)

```bash
pip install -r requirements.txt
```

## Usage

### Quick Test

Run the simple test to verify everything is working:

```bash
python simple_test.py
```

### Layer-by-Layer Inference

The main script `layer_by_layer_inference.py` provides a comprehensive interface for running Gemma 3 1B with access to every layer's activations.

```bash
python layer_by_layer_inference.py
```

## Key Features

### 1. Access Layer Activations

```python
from layer_by_layer_inference import LayerByLayerGemma

model = LayerByLayerGemma(model_id="google/gemma-3-1b-pt")
result = model.run_layer_by_layer("The capital of France is")

# Access activations from each layer
for layer_output in result['layer_outputs']:
    print(f"Layer {layer_output['layer_idx']}: {layer_output['hidden_states'].shape}")
```

### 2. Modify Activations

```python
def modify_activations(layer_idx, hidden_states):
    """Amplify activations at specific layers"""
    if layer_idx == 10:
        return hidden_states * 1.5
    return hidden_states

result = model.run_with_intermediate_modifications(
    "Hello world",
    modify_fn=modify_activations
)
```

### 3. Token-by-Token Generation

```python
generated = model.generate_token_by_token(
    "Once upon a time",
    max_new_tokens=10,
    temperature=0.7
)
```

## Model Architecture

- **Layers**: 26 transformer layers
- **Hidden Size**: 2304
- **Intermediate Size**: 9216
- **Attention Heads**: 8
- **Key-Value Heads**: 4 (Grouped Query Attention)
- **Context Length**: 128K tokens

## Files

- `layer_by_layer_inference.py` - Main script with LayerByLayerGemma class
- `simple_test.py` - Quick test to verify model loading
- `requirements.txt` - Python dependencies
- `venv/` - Virtual environment directory

## HuggingFace Access

You'll need to accept Google's license for Gemma models on HuggingFace:
1. Visit https://huggingface.co/google/gemma-3-1b-pt
2. Accept the license agreement
3. Login with `huggingface-cli login` if needed

## Model Variants

- `google/gemma-3-1b-pt` - Pretrained model
- `google/gemma-3-1b-it` - Instruction-tuned model

## Use Cases

This setup is perfect for:
- Analyzing internal representations
- Feature extraction and probing
- Activation steering and circuit analysis
- Linear feature map experiments
- Understanding model behavior at each layer
