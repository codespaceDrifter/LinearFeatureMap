import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./weights/phi4-mini", torch_dtype=torch.bfloat16, trust_remote_code=False)

print("RMSNorm gamma stats per layer\n")
print(f"{'Layer':<8} {'Norm':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print("=" * 70)

for i, layer in enumerate(model.model.layers):
    input_gamma = layer.input_layernorm.weight.data
    post_attn_gamma = layer.post_attention_layernorm.weight.data
    
    print(f"{i:<8} {'input_layernorm':<20} {input_gamma.mean():.4f}    {input_gamma.std():.4f}    {input_gamma.min():.4f}    {input_gamma.max():.4f}")
    print(f"{'':<8} {'post_attn_layernorm':<20} {post_attn_gamma.mean():.4f}    {post_attn_gamma.std():.4f}    {post_attn_gamma.min():.4f}    {post_attn_gamma.max():.4f}")

# final norm
final_gamma = model.model.norm.weight.data
print(f"\n{'final':<8} {'norm':<20} {final_gamma.mean():.4f}    {final_gamma.std():.4f}    {final_gamma.min():.4f}    {final_gamma.max():.4f}")
