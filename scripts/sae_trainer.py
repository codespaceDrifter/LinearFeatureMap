"""
SAE training - can be run standalone or imported.
"""
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from interp_algo.SAE import SAE
from scripts.config import config, pathconfig

BATCH_SIZE = 16384
LR = 1e-4
LAMBDA_SPARSE = 1
EPOCHS = 20


class SimpleDataset(Dataset):
    def __init__(self, path, total_tokens, embed_dim):
        self.data = np.memmap(path, dtype=np.float32, mode="r", shape=(total_tokens, embed_dim))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].copy())


def train_sae(data_path, save_path, total_tokens):
    """
    Train SAE on activation data.
    Returns the trained SAE (still on device).
    """
    print(f"Training SAE: {data_path} -> {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = SimpleDataset(data_path, total_tokens, config["embed_dim"])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    sae = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for batch_idx, x in enumerate(loader):
            x = x.to(config["device"])
            x_hat, z = sae(x)

            recon_loss = (x - x_hat).pow(2).mean()
            sparse_loss = z.mean()
            loss = recon_loss + LAMBDA_SPARSE * sparse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                any_active = (z > 0).float().mean().item() * config["hidden_dim"]
                r_squared = 1 - (x - x_hat).pow(2).sum() / x.pow(2).sum()
                print(f"  [{epoch+1}/{EPOCHS}][{batch_idx}/{len(loader)}] loss: {loss.item():.4f} R²: {r_squared.item():.4f} active: {any_active:.0f}")

        torch.save(sae.state_dict(), save_path)

    return sae


# standalone execution (old behavior - combined mlp+att)
if __name__ == "__main__":
    metadata = np.load(pathconfig["metadata"], allow_pickle=True).item()
    total_tokens = metadata["total_tokens"]

    for layer in range(config["num_layers"] - 1):
        # train mlp SAE
        train_sae(
            pathconfig["activations"]["mlp"][layer],
            pathconfig["sae"]["mlp"][layer],
            total_tokens
        )
        # train att SAE
        train_sae(
            pathconfig["activations"]["att"][layer + 1],
            pathconfig["sae"]["att"][layer + 1],
            total_tokens
        )