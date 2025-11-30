import torch
import torch.nn as nn
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
LR = 1e-4
LAMBDA_L1 = 1e-3
EPOCHS = 5
DEVICE = "cuda"

metadata = np.load("./data/activations/metadata.npy", allow_pickle=True).item()
total_tokens = metadata["total_tokens"]

def train_layer(layer):
    print(f"\n{'='*50}")
    print(f"Training LFM for layer {layer}")
    print(f"{'='*50}\n")
    
    # load data
    pre_data = np.memmap(f"./data/activations/layer_{layer}_pre.bin", dtype=np.float32, mode="r", shape=(total_tokens, EMBED_DIM))
    post_data = np.memmap(f"./data/activations/layer_{layer}_post.bin", dtype=np.float32, mode="r", shape=(total_tokens, EMBED_DIM))
    
    # load SAEs
    sae_pre = SAE(EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    sae_pre.load_state_dict(torch.load(f"./weights/SAE/layer_{layer}_pre_sae.pt"))
    sae_pre.eval()
    
    sae_post = SAE(EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    sae_post.load_state_dict(torch.load(f"./weights/SAE/layer_{layer}_post_sae.pt"))
    sae_post.eval()
    
    # init LFM
    lfm = LFM(HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(lfm.parameters(), lr=LR)
    
    num_batches = total_tokens // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch + 1}/{EPOCHS} ---")
        
        for batch_idx in range(num_batches):
            start = batch_idx * BATCH_SIZE
            end = start + BATCH_SIZE
            
            mlp_in = torch.from_numpy(pre_data[start:end].copy()).to(DEVICE)
            mlp_out = torch.from_numpy(post_data[start:end].copy()).to(DEVICE)
            
            with torch.no_grad():
                f_in = torch.relu(sae_pre.encoder(mlp_in))
                f_out = torch.relu(sae_post.encoder(mlp_out))
            
            f_pred = lfm(f_in)
            
            # loss 1: feature space
            loss_feature = (f_out - f_pred).pow(2).mean()
            
            # loss 2: reconstruction space
            mlp_out_pred = sae_post.decoder(f_pred)
            loss_recon = (mlp_out - mlp_out_pred).pow(2).mean()
            
            # loss 3: sparsity
            loss_l1 = lfm.linear.weight.abs().mean()
            
            loss = loss_feature + loss_recon + LAMBDA_L1 * loss_l1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"  [{batch_idx}/{num_batches}] loss: {loss.item():.2f} feature: {loss_feature.item():.2f} recon: {loss_recon.item():.2f} sparsity: {loss_l1.item():.2f}")
        
        torch.save(lfm.state_dict(), f"./weights/LFM/layer_{layer}_lfm.pt")
        print(f"  Saved epoch {epoch + 1}")
    
    print(f"Done with layer {layer}\n")

if __name__ == "__main__":
    os.makedirs("./weights/LFM", exist_ok=True)
    for layer in LAYERS:
        train_layer(layer)
    print("All done!")