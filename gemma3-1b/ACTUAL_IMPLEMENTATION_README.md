# Gemma 3 1B - ACTUAL PyTorch Implementation

This folder contains the **REAL** Google implementation of Gemma 3 1B, not a transformers wrapper.

## What You Have Here

### Official Google Code
The `gemma_pytorch/` directory contains Google's official PyTorch implementation:
- **`gemma_pytorch/gemma/model.py`** - The ACTUAL model code (RMSNorm, Attention, MLP, etc.)
- **`gemma_pytorch/gemma/config.py`** - Model configuration
- **`gemma_pytorch/gemma/tokenizer.py`** - Tokenizer implementation

### Your Scripts
- **`load_and_run.py`** - Shows you the model structure and execution flow
- **`layer_by_layer_inference.py`** - My wrapper for easy layer access (uses transformers)
- **`activation_analysis.py`** - Analysis tools

## The REAL Implementation

### Core Components (in gemma_pytorch/gemma/model.py)

#### 1. RMSNorm (lines 166-190)
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = True):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)
```

#### 2. GemmaMLP (lines 193-212)
```python
class GemmaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, quant: bool):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, quant)
        self.up_proj = Linear(hidden_size, intermediate_size, quant)
        self.down_proj = Linear(intermediate_size, hidden_size, quant)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")  # PytorchGELUTanh
        up = self.up_proj(x)
        fuse = gate * up  # Gated activation
        outputs = self.down_proj(fuse)
        return outputs
```

#### 3. GemmaAttention (lines 215-339)
- Grouped-Query Attention (8 Q heads, 2 KV heads)
- QK normalization (new in Gemma 3, replaces soft-capping from Gemma 2)
- Rotary Position Embeddings (RoPE)
- Local sliding window or global attention

Key parts:
```python
# QKV projection
qkv = self.qkv_proj(hidden_states)
xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

# QK normalization (Gemma 3 specific)
if self.query_norm is not None and self.key_norm is not None:
    xq = self.query_norm(xq)
    xk = self.key_norm(xk)

# Apply RoPE
xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

# Grouped-query attention (repeat KV heads)
key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

# Attention scores
scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
scores = F.softmax(scores, dim=-1)
output = torch.matmul(scores, v)
```

#### 4. Gemma2DecoderLayer (lines 394-458)
Used for both Gemma 2 and Gemma 3 (Gemma 3 uses Gemma2DecoderLayer):
```python
class Gemma2DecoderLayer(nn.Module):
    def forward(self, hidden_states, freqs_cis, kv_cache, mask, local_mask):
        # Self Attention with pre-norm and post-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(...)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with pre-norm and post-norm (Gemma 3 specific)
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)  # Gemma 3
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)  # Gemma 3
        hidden_states = residual + hidden_states

        return hidden_states
```

#### 5. GemmaModel (lines 461-507)
The full transformer stack:
```python
class GemmaModel(nn.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            attn_type = config.attn_types[i % len(config.attn_types)]
            self.layers.append(Gemma2DecoderLayer(config, attn_type))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, freqs_cis, kv_caches, mask, local_mask):
        for layer in self.layers:
            hidden_states = layer(...)
        hidden_states = self.norm(hidden_states)
        return hidden_states
```

## Gemma 3 1B Architecture

### Specifications
- **Layers**: 26 decoder layers
- **Hidden size**: 1152
- **Intermediate size**: 6912 (MLP)
- **Attention heads**: 8 query heads
- **Key-Value heads**: 2 (Grouped Query Attention - 4:1 ratio)
- **Head dimension**: 128
- **Vocab size**: 256,000
- **Max context**: 32,768 tokens
- **Sliding window**: 1,024 tokens

### Attention Pattern
Gemma 3 alternates between local and global attention:
- Pattern: **5 local + 1 global** (repeating)
- Layers 0-4: Local sliding window (1024 tokens)
- Layer 5: Global attention
- Layers 6-10: Local sliding window
- Layer 11: Global attention
- ... continues for all 26 layers

### Key Innovations

#### 1. QK Normalization (replaces soft-capping)
```python
# Before attention, normalize Q and K
xq = self.query_norm(xq)  # RMSNorm on queries
xk = self.key_norm(xk)    # RMSNorm on keys
```

#### 2. Four RMSNorms per layer
Unlike Gemma 1 (2 norms) or standard transformers:
- `input_layernorm` - Before attention
- `post_attention_layernorm` - After attention
- `pre_feedforward_layernorm` - Before MLP
- `post_feedforward_layernorm` - After MLP

#### 3. Grouped Query Attention
8 query heads share 2 key-value heads:
```python
# Repeat KV heads to match Q heads
key = torch.repeat_interleave(key, num_queries_per_kv=4, dim=2)
value = torch.repeat_interleave(value, num_queries_per_kv=4, dim=2)
```

This reduces memory and computation while maintaining quality.

## How To Use

### 1. Run the demo
```bash
cd gemma3-1b
source venv/bin/activate
python load_and_run.py
```

This will show you the model structure and execution flow.

### 2. Read the actual code
Open `gemma_pytorch/gemma/model.py` and read through:
- Line 166: RMSNorm
- Line 193: GemmaMLP
- Line 215: GemmaAttention
- Line 394: Gemma2DecoderLayer (used for Gemma 3)
- Line 461: GemmaModel

### 3. Modify and experiment
Since this is the actual PyTorch code, you can:
- Add print statements to see activations
- Modify attention mechanisms
- Extract intermediate layer outputs
- Implement custom interventions

## Loading Weights

To actually run the model, you need to download weights from:
1. **Hugging Face**: https://huggingface.co/google/gemma-3-1b-pt
2. **Kaggle**: https://www.kaggle.com/models/google/gemma-3

Then use the Google implementation's loading method or convert to the format expected by this code.

## Why This Matters

This is NOT a black box. You have:
1. ✅ The actual attention mechanism code
2. ✅ The actual MLP gating logic
3. ✅ The actual normalization layers
4. ✅ The actual RoPE implementation
5. ✅ Full control to modify anything

You can literally add a `print()` statement anywhere to see what's happening, or modify the forward pass to capture activations, or change how attention works.

## Files Map

```
gemma3-1b/
├── gemma_pytorch/              # Official Google implementation
│   └── gemma/
│       ├── model.py           # ⭐ THE REAL MODEL CODE
│       ├── config.py          # Configuration classes
│       ├── tokenizer.py       # Tokenizer
│       └── gemma3_model.py    # Multimodal wrapper (not needed for text)
│
├── load_and_run.py            # Demo showing model structure
├── layer_by_layer_inference.py # Easy wrapper (uses transformers)
├── activation_analysis.py     # Analysis tools
├── simple_test.py             # Quick test
├── requirements.txt           # Dependencies
└── README.md                  # General usage

```

## Next Steps

1. ✅ You have the actual implementation
2. ⏳ Download model weights
3. ⏳ Load weights and run inference
4. ⏳ Add your own layer-by-layer hooks
5. ⏳ Implement linear feature maps or SAEs

**The code is right here. It's real PyTorch. Go explore it!**
