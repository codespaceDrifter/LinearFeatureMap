import torch
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from SAE.SAE import SAE

LAYERS = [8, 16, 24, 31]
EMBED_DIM = 3072
EXPANSION = 8
HIDDEN_DIM = EMBED_DIM * EXPANSION
BATCH_SIZE = 4096
DEVICE = "cuda"

metadata = np.load("./data/test/activations/metadata.npy", allow_pickle=True).item()
total_tokens = metadata["total_tokens"]

def eval_sae(layer, pre_post):
    data_path = f"./data/test/activations/layer_{layer}_{pre_post}.bin"
    weight_path = f"./weights/SAE/layer_{layer}_{pre_post}_sae.pt"
    
    if not os.path.exists(weight_path):
        print(f"Skipping {layer} {pre_post} - no weights found")
        return
    
    data = np.memmap(data_path, dtype=np.float32, mode="r", shape=(total_tokens, EMBED_DIM))
    
    sae = SAE(EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    sae.load_state_dict(torch.load(weight_path))
    sae.eval()
    
    total_recon = 0
    total_norm = 0
    total_active = 0
    total_mid = 0
    total_strong = 0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, total_tokens, BATCH_SIZE):
            end = min(i + BATCH_SIZE, total_tokens)
            x = torch.from_numpy(data[i:end].copy()).to(DEVICE)
            
            x_hat, z = sae(x)
            
            total_recon += (x - x_hat).pow(2).sum().item()
            total_norm += x.pow(2).sum().item()
            total_active += (z > 0).float().sum().item()
            total_mid += (z > 0.5).float().sum().item()
            total_strong += (z > 1).float().sum().item()
            num_batches += x.shape[0]
    
    r2 = 1 - total_recon / total_norm
    avg_active = total_active / num_batches
    avg_mid = total_mid / num_batches
    avg_strong = total_strong / num_batches
    
    print(f"Layer {layer} {pre_post}:")
    print(f"  R²: {r2:.4f}")
    print(f"  Active (>0): {avg_active:.1f}")
    print(f"  Mid (>0.5): {avg_mid:.1f}")
    print(f"  Strong (>1): {avg_strong:.1f}")
    print()

if __name__ == "__main__":
    print(f"Evaluating on {total_tokens} test tokens\n")
    for layer in LAYERS:
        eval_sae(layer, "pre")
        eval_sae(layer, "post")