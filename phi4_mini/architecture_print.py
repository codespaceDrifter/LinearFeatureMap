# prints the phi4-mini architecture

from transformers import AutoModelForCausalLM
import torch
import inspect
import os

model = AutoModelForCausalLM.from_pretrained("./weights/phi4-mini", dtype=torch.bfloat16, device_map="auto", trust_remote_code=False)

os.makedirs("phi4-mini/architecture", exist_ok=True)

layer = model.model.layers[0]

classes = {
    "CausalLM": model.__class__,
    "Model": model.model.__class__,
    "DecoderLayer": layer.__class__,
    "Attention": layer.self_attn.__class__,
    "MLP": layer.mlp.__class__,
    "RMSNorm": layer.input_layernorm.__class__,
    "RotaryEmbedding": model.model.rotary_emb.__class__,
}

if hasattr(layer.mlp, 'activation_fn'):
    classes["Activation"] = layer.mlp.activation_fn.__class__

for name, cls in classes.items():
    try:
        source = inspect.getsource(cls)
        with open(f"phi4-mini/architecture/{name}.py", "w") as f:
            f.write(source)
        print(f"✓ {name}.py ({cls.__name__})")
    except Exception as e:
        print(f"✗ {name}: {e}")

# pretty print
print("\n" + "="*60)
print("=== PHI-4-MINI ARCHITECTURE ===")
print("="*60)

embed = model.model.embed_tokens
n_layers = len(model.model.layers)
mlp = layer.mlp
attn = layer.self_attn

print(f"""
{model.__class__.__name__}
├─ {model.model.__class__.__name__}
│  ├─ embed_tokens: ({embed.num_embeddings}, {embed.embedding_dim})
│  ├─ layers: {n_layers} x {layer.__class__.__name__}
│  │  ├─ input_layernorm: ({layer.input_layernorm.weight.shape[0]})
│  │  ├─ self_attn: {attn.__class__.__name__}
│  │  │  ├─ qkv_proj: ({attn.qkv_proj.in_features} → {attn.qkv_proj.out_features})
│  │  │  └─ o_proj: ({attn.o_proj.in_features} → {attn.o_proj.out_features})
│  │  ├─ post_attention_layernorm: ({layer.post_attention_layernorm.weight.shape[0]})
│  │  └─ mlp: {mlp.__class__.__name__}
│  │     ├─ gate_up_proj: ({mlp.gate_up_proj.in_features} → {mlp.gate_up_proj.out_features})
│  │     ├─ activation: {mlp.activation_fn.__class__.__name__}
│  │     └─ down_proj: ({mlp.down_proj.in_features} → {mlp.down_proj.out_features})
│  ├─ norm: ({model.model.norm.weight.shape[0]})
│  └─ rotary_emb: {model.model.rotary_emb.__class__.__name__}
└─ lm_head: ({model.lm_head.in_features} → {model.lm_head.out_features})
""")