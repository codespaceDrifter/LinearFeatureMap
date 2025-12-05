import torch
import torch.nn as nn
import numpy as np
import os
from interp_algo.SAE import SAE
from scripts.config import config, pathconfig

BATCH_SIZE = 16384  # large batch for SAE training
LR = 1e-4
LAMBDA_SPARSE = 1
EPOCHS = 20

os.makedirs("./weights/SAE", exist_ok=True)

metadata = np.load(pathconfig["metadata"], allow_pickle=True).item()
total_tokens = metadata["total_tokens"]

class CombinedDataset(torch.utils.data.Dataset):
    """Combined dataset: first half is mlp_in, second half is att_in"""
    def __init__(self, mlp_in_path, att_in_path, num_tokens, embed_dim):
        self.mlp_in = np.memmap(mlp_in_path, dtype=np.float32, mode="r", shape=(num_tokens, embed_dim))
        self.att_in = np.memmap(att_in_path, dtype=np.float32, mode="r", shape=(num_tokens, embed_dim))
        self.num_tokens = num_tokens
        self.total_len = num_tokens * 2  # combined length
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < self.num_tokens:
            return torch.from_numpy(self.mlp_in[idx].copy())  # (3072,)
        else:
            return torch.from_numpy(self.att_in[idx - self.num_tokens].copy())  # (3072,)

def train_layer(layer):
    print(f"\n{'='*50}")
    print(f"Training SAE for layer {layer} (mlp_in + layer {layer+1} att_in)")
    print(f"{'='*50}\n")

    dataset = CombinedDataset(
        pathconfig["activations"][layer]["mlp"],
        pathconfig["activations"][layer]["att"],
        total_tokens,
        config["embed_dim"]
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    sae = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch + 1}/{EPOCHS} ---")
        
        for batch_idx, x in enumerate(loader):
            x = x.to(config["device"])  # (batch, 3072)
            
            x_hat, z = sae(x)  # x_hat: (batch, 3072), z: (batch, hidden_dim)
            
            recon_loss = (x - x_hat).pow(2).mean()
            sparse_loss = z.mean()
            loss = recon_loss + LAMBDA_SPARSE * sparse_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 50 == 0:
                any_active = (z > 0).float().mean().item() * config["hidden_dim"]
                mid_active = (z > 0.5).float().mean().item() * config["hidden_dim"]
                big_active = (z > 1).float().mean().item() * config["hidden_dim"]
                r_squared = 1 - (x - x_hat).pow(2).sum() / x.pow(2).sum()
                print(f"  [{batch_idx}/{len(loader)}] loss: {loss.item():.4f} R²: {r_squared.item():.4f} active>0: {any_active:.0f} active>0.5: {mid_active:.0f} active>1: {big_active:.0f}")
        
        torch.save(sae.state_dict(), pathconfig["sae"][layer])
        print(f"  Saved epoch {epoch + 1}")

    print(f"Done with layer {layer}\n")

if __name__ == "__main__":
    for layer in config["layers"]:
        train_layer(layer)
    print("All done!")