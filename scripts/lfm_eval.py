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
    
    total_feature = 0
    total_recon = 0
    total_l1 = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx in range(total_tokens // BATCH_SIZE):
            start = batch_idx * BATCH_SIZE
            end = start + BATCH_SIZE
            
            mlp_in = torch.from_numpy(pre_data[start:end].copy()).to(DEVICE)
            mlp_out = torch.from_numpy(post_data[start:end].copy()).to(DEVICE)
            
            f_in = torch.relu(sae_pre.encoder(mlp_in))
            f_out = torch.relu(sae_post.encoder(mlp_out))
            
            f_pred = lfm(f_in)
            
            total_feature += (f_out - f_pred).pow(2).mean().item()
            
            mlp_out_pred = sae_post.decoder(f_pred)
            total_recon += (mlp_out - mlp_out_pred).pow(2).mean().item()
            
            total_l1 += lfm.linear.weight.abs().mean().item()
            
            num_batches += 1
    
    avg_feature = total_feature / num_batches
    avg_recon = total_recon / num_batches
    avg_l1 = total_l1 / num_batches
    avg_total = avg_feature + avg_recon + avg_l1
    
    print(f"Feature loss: {avg_feature:.6f}")
    print(f"Recon loss:   {avg_recon:.6f}")
    print(f"L1 loss:      {avg_l1:.6f}")
    print(f"Total loss:   {avg_total:.6f}")

if __name__ == "__main__":
    for layer in LAYERS:
        eval_layer(layer)
    print("\nDone!")