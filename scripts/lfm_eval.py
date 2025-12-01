import torch
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from interp_algo.SAE import SAE
from interp_algo.LFM import LFM

LAYERS = [8, 16, 24]
EMBED_DIM = 3072
HIDDEN_DIM = 12288
BATCH_SIZE = 4096
ACTIVE_THRESHOLD = 0.2
DEVICE = "cuda"

metadata = np.load("./data/test/activations/metadata.npy", allow_pickle=True).item()
total_tokens = metadata["total_tokens"]

def eval_layer(layer):
    print(f"\n{'='*50}")
    print(f"Evaluating LFM for layer {layer}")
    print(f"{'='*50}\n")
    
    pre_data = np.memmap(f"./data/test/activations/layer_{layer}_pre.bin", dtype=np.float32, mode="r", shape=(total_tokens, EMBED_DIM))
    post_data = np.memmap(f"./data/test/activations/layer_{layer}_post.bin", dtype=np.float32, mode="r", shape=(total_tokens, EMBED_DIM))
    
    sae_pre = SAE(EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    sae_pre.load_state_dict(torch.load(f"./weights/SAE/layer_{layer}_pre_sae.pt"))
    sae_pre.eval()
    
    sae_post = SAE(EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    sae_post.load_state_dict(torch.load(f"./weights/SAE/layer_{layer}_post_sae.pt"))
    sae_post.eval()
    
    lfm = LFM(HIDDEN_DIM).to(DEVICE)
    lfm.load_state_dict(torch.load(f"./weights/LFM/layer_{layer}_lfm.pt"))
    lfm.eval()
    
    # unmasked accumulators
    total_feature = 0
    total_recon = 0
    
    # masked accumulators
    masked_feature_sum = 0
    masked_feature_count = 0
    masked_recon_sum = 0
    masked_recon_count = 0
    
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx in range(total_tokens // BATCH_SIZE):
            start = batch_idx * BATCH_SIZE
            end = start + BATCH_SIZE
            
            mlp_in = torch.from_numpy(pre_data[start:end].copy()).to(DEVICE)
            mlp_out = torch.from_numpy(post_data[start:end].copy()).to(DEVICE)
            
            f_in = torch.relu(sae_pre.encoder(mlp_in))
            f_out = torch.relu(sae_post.encoder(mlp_out))
            
            f_in_masked = f_in * (f_in > ACTIVE_THRESHOLD)
            f_out_masked = f_out * (f_out > ACTIVE_THRESHOLD)
            
            f_pred = lfm(f_in_masked)
            mlp_out_pred = sae_post.decoder(f_pred)
            
            # unmasked
            total_feature += (f_out - f_pred).pow(2).mean().item()
            total_recon += (mlp_out - mlp_out_pred).pow(2).mean().item()
            
            # masked
            active_mask = f_out > ACTIVE_THRESHOLD
            if active_mask.any():
                rel_error = (f_out_masked - f_pred) / (f_out_masked + 1e-6)
                masked_feature_sum += (rel_error[active_mask]).pow(2).sum().item()
                masked_feature_count += active_mask.sum().item()
            masked_recon_sum += (mlp_out - mlp_out_pred).pow(2).sum().item()
            masked_recon_count += mlp_out.numel()
            
            num_batches += 1
    
    print("--- Unmasked (all features) ---")
    print(f"  Feature MSE: {total_feature / num_batches:.6f}")
    print(f"  Recon MSE:   {total_recon / num_batches:.6f}")
    
    print(f"\n--- Masked (f_out > {ACTIVE_THRESHOLD}) ---")
    avg_rel_error = (masked_feature_sum / masked_feature_count) ** 0.5 * 100
    print(f"  Avg Rel Error: {avg_rel_error:.1f}%")
    print(f"  Recon MSE:     {masked_recon_sum / masked_recon_count:.6f}")
    
    print(f"\nWeight distribution:")
    weights = lfm.linear.weight.data.abs().flatten()
    total_weights = weights.numel()
    
    lower = 0.0
    while True:
        upper = lower + 0.1
        count = ((weights >= lower) & (weights < upper)).sum().item()
        pct = 100 * count / total_weights
        print(f"  {lower:.1f}-{upper:.1f}: {pct:.4f}%")
        if (weights >= upper).sum() == 0:
            break
        lower = upper

if __name__ == "__main__":
    for layer in LAYERS:
        eval_layer(layer)
    print("\nDone!")