"""
Create a MergedSAE from all individual SAEs for efficient TUI encoding.

Structure: 64 hook positions (32 layers x 2 positions each)
  - hook_idx = layer * 2     -> att_in[layer]
  - hook_idx = layer * 2 + 1 -> mlp_in[layer]

We have SAEs for: mlp[0-30] + att[1-31] (62 total)
Missing: att[0] (hook_idx=0) and mlp[31] (hook_idx=63) - left as zeros
"""
import torch
import os
from scripts.config import config, pathconfig
from interp_algo.SAE import SAE
from interp_algo.mergedSAE import MergedSAE

num_layers = config["num_layers"]
embed_dim = config["embed_dim"]
hidden_dim = config["hidden_dim"]

num_hooks = num_layers * 2  # 64 hook positions

print(f"Creating MergedSAE with {num_hooks} hook positions...")

# create merged SAE
merged = MergedSAE(num_hooks, embed_dim, hidden_dim)

loaded = 0

# load mlp SAEs (layers 0-30) -> hook indices 1, 3, 5, ..., 61
print("Loading mlp SAEs...")
for layer in range(num_layers - 1):  # 0 to 30
    path = pathconfig["sae"]["mlp"][layer]
    if not os.path.exists(path):
        print(f"  [skip] mlp[{layer}] - not found")
        continue

    sae = SAE(embed_dim, hidden_dim)
    sae.load_state_dict(torch.load(path, weights_only=True))

    hook_idx = layer * 2 + 1

    # encoder: Linear(embed_dim, hidden_dim) -> weight is (hidden_dim, embed_dim)
    # we need (embed_dim, hidden_dim) for bmm
    merged.encoder_weight.data[hook_idx] = sae.encoder.weight.data.T
    merged.encoder_bias.data[hook_idx] = sae.encoder.bias.data

    # decoder: Linear(hidden_dim, embed_dim) -> weight is (embed_dim, hidden_dim)
    # we need (hidden_dim, embed_dim) for bmm
    merged.decoder_weight.data[hook_idx] = sae.decoder.weight.data.T
    merged.decoder_bias.data[hook_idx] = sae.decoder.bias.data

    loaded += 1
    print(f"  mlp[{layer}] -> hook {hook_idx}")

# load att SAEs (layers 1-31) -> hook indices 2, 4, 6, ..., 62
print("Loading att SAEs...")
for layer in range(1, num_layers):  # 1 to 31
    path = pathconfig["sae"]["att"][layer]
    if not os.path.exists(path):
        print(f"  [skip] att[{layer}] - not found")
        continue

    sae = SAE(embed_dim, hidden_dim)
    sae.load_state_dict(torch.load(path, weights_only=True))

    hook_idx = layer * 2

    merged.encoder_weight.data[hook_idx] = sae.encoder.weight.data.T
    merged.encoder_bias.data[hook_idx] = sae.encoder.bias.data
    merged.decoder_weight.data[hook_idx] = sae.decoder.weight.data.T
    merged.decoder_bias.data[hook_idx] = sae.decoder.bias.data

    loaded += 1
    print(f"  att[{layer}] -> hook {hook_idx}")

print(f"\nLoaded {loaded}/62 SAEs")
print(f"Missing hook positions: 0 (att[0]), 63 (mlp[31]) - left as zeros")

print(f"\nMerged encoder weight shape: {merged.encoder_weight.shape}")
print(f"Merged decoder weight shape: {merged.decoder_weight.shape}")

# save
os.makedirs(os.path.dirname(pathconfig["merged_sae"]), exist_ok=True)
torch.save(merged.state_dict(), pathconfig["merged_sae"])
print(f"\nSaved to {pathconfig['merged_sae']}")
