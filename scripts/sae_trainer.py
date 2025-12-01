import torch
import torch.nn as nn
import numpy as np
import os
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
sys.path.append(str(Path(__file__).parent.parent))
from interp_algo.SAE import SAE

LAYERS = [8, 16, 24]
EMBED_DIM = 3072
EXPANSION = 4
HIDDEN_DIM = EMBED_DIM * EXPANSION
BATCH_SIZE = 8192
LR = 1e-4
LAMBDA_SPARSE = 1
EPOCHS = 20
DEVICE = "cuda"

os.makedirs("./weights/SAE", exist_ok=True)

metadata = np.load("./data/activations/metadata.npy", allow_pickle=True).item()
total_tokens = metadata["total_tokens"]

class MemmapDataset(Dataset):
    def __init__(self, pre_path, post_path, total_tokens, embed_dim):
        self.pre = np.memmap(pre_path, dtype=np.float32, mode="r", shape=(total_tokens, embed_dim))
        self.post = np.memmap(post_path, dtype=np.float32, mode="r", shape=(total_tokens, embed_dim))
        self.total = total_tokens
    
    def __len__(self):
        return self.total * 2
    
    def __getitem__(self, idx):
        if idx < self.total:
            return torch.from_numpy(self.pre[idx].copy())
        else:
            return torch.from_numpy(self.post[idx - self.total].copy())

def train_layer(layer):
    print(f"\n{'='*50}")
    print(f"Training unified SAE for layer {layer}")
    print(f"{'='*50}\n")
    
    dataset = MemmapDataset(
        f"./data/activations/layer_{layer}_pre.bin",
        f"./data/activations/layer_{layer}_post.bin",
        total_tokens, EMBED_DIM
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    sae = SAE(EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch + 1}/{EPOCHS} ---")
        
        for batch_idx, x in enumerate(loader):
            x = x.to(DEVICE)
            
            x_hat, z = sae(x)
            
            recon_loss = (x - x_hat).pow(2).mean()
            sparse_loss = z.mean()
            loss = recon_loss + LAMBDA_SPARSE * sparse_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                any_active = (z > 0).float().mean().item() * HIDDEN_DIM
                mid_active = (z > 0.5).float().mean().item() * HIDDEN_DIM
                big_active = (z > 1).float().mean().item() * HIDDEN_DIM
                r_squared = 1 - (x - x_hat).pow(2).sum() / x.pow(2).sum()
                print(f"  [{batch_idx}/{len(loader)}] loss: {loss.item():.4f} R²: {r_squared.item():.4f} active>0: {any_active:.0f} >0.5: {mid_active:.0f} >1: {big_active:.0f}")
        
        torch.save(sae.state_dict(), f"./weights/SAE/layer_{layer}_sae.pt")
        print(f"  Saved epoch {epoch + 1}")
    
    print(f"Done with layer {layer}\n")

if __name__ == "__main__":
    for layer in LAYERS:
        train_layer(layer)
    print("All done!")