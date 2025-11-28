import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append(".")
from SAE.SAE import SAE

# config
EMBED_DIM = 3072
EXPANSION = 4
HIDDEN_DIM = EMBED_DIM * EXPANSION
BATCH_SIZE = 16384
LR = 1e-4
LAMBDA_SPARSE = 1
EPOCHS = 20
DEVICE = "cuda"

os.makedirs("./weights/SAE", exist_ok=True)

metadata = np.load("./data/activations/metadata.npy", allow_pickle=True).item()
total_tokens = metadata["total_tokens"]

def train_layer(layer, pre_post):
    print(f"\n{'='*50}")
    print(f"Training layer {layer} {pre_post}")
    print(f"{'='*50}\n")
    
    data_path = f"./data/activations/layer_{layer}_{pre_post}.bin"
    save_path = f"./weights/SAE/layer_{layer}_{pre_post}_sae.pt"
    
    data = np.memmap(data_path, dtype=np.float32, mode="r", shape=(total_tokens, EMBED_DIM))
    
    sae = SAE(EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
    
    num_batches = total_tokens // BATCH_SIZE
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch + 1}/{EPOCHS} ---")
        
        for batch_idx in range(num_batches):
            start = batch_idx * BATCH_SIZE
            end = start + BATCH_SIZE
            
            x = torch.from_numpy(data[start:end].copy()).to(DEVICE)
            
            x_hat, z = sae(x)
            
            recon_loss = (x - x_hat).pow(2).mean()
            sparse_loss = z.mean()
            loss = recon_loss + LAMBDA_SPARSE * sparse_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 50 == 0:
                any_active = (z > 0).float().mean().item() * HIDDEN_DIM
                small_active = (z > 0.1).float().mean().item() * HIDDEN_DIM
                mid_active = (z > 0.5).float().mean().item() * HIDDEN_DIM
                big_active = (z > 1).float().mean().item() * HIDDEN_DIM
                r_squared = 1 - (x - x_hat).pow(2).sum() / x.pow(2).sum()
                print(f"  [{batch_idx}/{num_batches}] loss: {loss.item():.4f} R²: {r_squared.item():.4f} any active: {any_active:.0f} small_active: {small_active: .0f} mid active: {mid_active:.0f} big active: {big_active:.0f}")
        
        torch.save(sae.state_dict(), save_path)
        print(f"  Saved epoch {epoch + 1} to {save_path}")
    

    print(f"Done with layer {layer} {pre_post}\n")

if __name__ == "__main__":
    LAYERS = [8, 16, 24, 31]
    
    for layer in LAYERS:
        train_layer(layer, "pre")
        train_layer(layer, "post")
    
    print("All done!")
