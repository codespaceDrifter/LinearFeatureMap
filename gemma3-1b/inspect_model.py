"""
Demo: What can AutoModelForCausalLM do?

This shows you can access EVERYTHING:
- Architecture
- Weights
- Activations
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# ===== 1. ARCHITECTURE =====
print("=" * 60)
print("1. READ ARCHITECTURE (without downloading model)")
print("=" * 60)

config = AutoConfig.from_pretrained("./weights")
print(f"\nconfig.json contains:")
print(f"  - num_hidden_layers: {config.num_hidden_layers}")
print(f"  - hidden_size: {config.hidden_size}")
print(f"  - intermediate_size: {config.intermediate_size}")
print(f"  - num_attention_heads: {config.num_attention_heads}")
print(f"  - num_key_value_heads: {config.num_key_value_heads}")
print(f"  - vocab_size: {config.vocab_size}")
print(f"  - Everything else: {list(config.to_dict().keys())}")

# ===== 2. WEIGHTS =====
print("\n" + "=" * 60)
print("2. READ WEIGHTS")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    "./weights",
    torch_dtype=torch.bfloat16,
    device_map="cpu"  # Use CPU for inspection
)

print(f"\nModel is a PyTorch nn.Module: {isinstance(model, torch.nn.Module)}")
print(f"\nYou can access:")
print(f"  - model.state_dict().keys() -> All parameter names")
print(f"  - model.model.layers[0] -> First transformer layer")
print(f"  - model.model.layers[0].mlp -> MLP module")
print(f"  - model.model.layers[0].mlp.gate_proj.weight -> Specific tensor")

# Show structure
print(f"\nModel structure:")
print(f"  model")
print(f"    ├── model.embed_tokens (Embedding)")
print(f"    ├── model.layers (ModuleList of {config.num_hidden_layers} layers)")
print(f"    │     └── layers[i]")
print(f"    │           ├── self_attn (Gemma2Attention)")
print(f"    │           │     ├── q_proj, k_proj, v_proj, o_proj")
print(f"    │           │     └── weights are [hidden_size, hidden_size] tensors")
print(f"    │           ├── mlp (Gemma2MLP)")
print(f"    │           │     ├── gate_proj [hidden_size, intermediate_size]")
print(f"    │           │     ├── up_proj [hidden_size, intermediate_size]")
print(f"    │           │     └── down_proj [intermediate_size, hidden_size]")
print(f"    │           ├── input_layernorm")
print(f"    │           └── post_attention_layernorm")
print(f"    ├── model.norm (final RMSNorm)")
print(f"    └── lm_head (Linear) [hidden_size, vocab_size]")

# Access specific weight
gate_proj_weight = model.model.layers[0].mlp.gate_proj.weight
print(f"\nExample - Layer 0 MLP gate_proj.weight:")
print(f"  Shape: {gate_proj_weight.shape}")  # [intermediate_size, hidden_size]
print(f"  Dtype: {gate_proj_weight.dtype}")
print(f"  Device: {gate_proj_weight.device}")
print(f"  First 5 values: {gate_proj_weight[0, :5]}")

# ===== 3. ACTIVATIONS =====
print("\n" + "=" * 60)
print("3. READ ACTIVATIONS")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained("./weights")
text = "The meaning of life is"
inputs = tokenizer(text, return_tensors="pt")

print(f"\nInput: '{text}'")
print(f"Tokens: {inputs['input_ids'][0].tolist()}")

# Method 1: output_hidden_states (gets all layer outputs)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

print(f"\nMethod 1: output_hidden_states=True")
print(f"  - outputs.hidden_states -> Tuple of {len(outputs.hidden_states)} tensors")
print(f"  - hidden_states[0] = embeddings: {outputs.hidden_states[0].shape}")
print(f"  - hidden_states[1] = after layer 0: {outputs.hidden_states[1].shape}")
print(f"  - hidden_states[26] = after layer 25: {outputs.hidden_states[26].shape}")

# Method 2: Forward hooks (gets specific module I/O)
print(f"\nMethod 2: Forward hooks (see extract_activations.py)")
print(f"  - model.model.layers[0].mlp.register_forward_hook(hook_fn)")
print(f"  - Hook captures PURE MLP input and output")
print(f"  - This is what we use for SAE training")

# ===== 4. WHAT'S IN ./weights FOLDER? =====
print("\n" + "=" * 60)
print("4. WHAT'S IN ./weights FOLDER?")
print("=" * 60)

print("""
When you download with HuggingFace, ./weights contains:

├── config.json              <- Architecture specification
├── model.safetensors        <- All weights in SafeTensors format
│   (or pytorch_model.bin)   <- (older format)
├── tokenizer.json           <- Tokenizer vocabulary and merges
├── tokenizer_config.json    <- Tokenizer settings
└── special_tokens_map.json  <- Special tokens (<bos>, <eos>, etc)

The .safetensors or .bin file contains:
- model.embed_tokens.weight
- model.layers.0.self_attn.q_proj.weight
- model.layers.0.self_attn.k_proj.weight
- model.layers.0.self_attn.v_proj.weight
- model.layers.0.self_attn.o_proj.weight
- model.layers.0.mlp.gate_proj.weight
- model.layers.0.mlp.up_proj.weight
- model.layers.0.mlp.down_proj.weight
- ... (repeat for all 26 layers)
- model.norm.weight
- lm_head.weight

Total: ~1B parameters × 2 bytes (bfloat16) = ~2GB
""")

print("\n" + "=" * 60)
print("SUMMARY: YES, YOU CAN ACCESS EVERYTHING")
print("=" * 60)
print("""
✓ Architecture: config.json (loaded automatically)
✓ Weights: All parameters via model.state_dict() or model.model.layers[i]
✓ Activations: Forward hooks or output_hidden_states=True

AutoModelForCausalLM is just a convenience wrapper that:
1. Reads config.json to know what architecture to build
2. Instantiates the PyTorch model (GemmaForCausalLM)
3. Loads weights from .safetensors into the model
4. Returns a full PyTorch nn.Module you can inspect/modify

The gemma_pytorch code shows you the SAME implementation.
Transformers just wraps it nicely.
""")
