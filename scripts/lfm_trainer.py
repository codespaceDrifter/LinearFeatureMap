import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from interp_algo.SAE import SAE
from interp_algo.LFM import LFM
from scripts.config import config, pathconfig

BATCH_SIZE = 4096  # LFM training batch
LR = 1e-4
LAMBDA_RECON = 0.1
ACTIVE_THRESHOLD = 0.1  # threshold for masked loss
EPOCHS = 5

os.makedirs("./weights/LFM", exist_ok=True)

metadata = np.load(pathconfig["metadata"], allow_pickle=True).item()
total_tokens = metadata["total_tokens"]

class PairedDataset(Dataset):
    """Loads mlp_in and next layer's att_in as paired data"""
    def __init__(self, mlp_in_path, att_in_path, total_tokens, embed_dim):
        self.mlp_in = np.memmap(mlp_in_path, dtype=np.float32, mode="r", shape=(total_tokens, embed_dim))
        self.att_in = np.memmap(att_in_path, dtype=np.float32, mode="r", shape=(total_tokens, embed_dim))
        self.total = total_tokens
    
    def __len__(self):
        return self.total
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.mlp_in[idx].copy()), torch.from_numpy(self.att_in[idx].copy())

def train_layer(layer):
    print(f"\n{'='*50}")
    print(f"Training LFM for layer {layer}")
    print(f"{'='*50}\n")

    dataset = PairedDataset(
        pathconfig["activations"][layer]["mlp"],
        pathconfig["activations"][layer]["att"],
        total_tokens, config["embed_dim"]
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # same SAE for both sides
    sae = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
    sae.load_state_dict(torch.load(pathconfig["sae"][layer]))
    sae.eval()

    lfm = LFM(config["hidden_dim"]).to(config["device"])
    optimizer = torch.optim.Adam(lfm.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch + 1}/{EPOCHS} ---")
        
        for batch_idx, (mlp_in, att_in) in enumerate(loader):
            mlp_in, att_in = mlp_in.to(config["device"]), att_in.to(config["device"])
            
            with torch.no_grad():
                f_in = sae.encode(mlp_in)  # (batch, hidden_dim)
                f_out = sae.encode(att_in)  # (batch, hidden_dim)
            
            f_pred = lfm(f_in)
            
            # masked feature loss - only where both in and out are active
            active_mask = (f_in > ACTIVE_THRESHOLD) & (f_out > ACTIVE_THRESHOLD)
            if active_mask.any():
                loss_feature = ((f_out - f_pred)[active_mask]).pow(2).mean()
            else:
                loss_feature = torch.tensor(0.0, device=config["device"])
            
            # recon loss (downweighted)
            att_in_pred = sae.decode(f_pred)
            loss_recon = (att_in - att_in_pred).pow(2).mean()
            
            loss = loss_feature + LAMBDA_RECON * loss_recon
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                n_active = active_mask.sum().item()
                print(f"  [{batch_idx}/{len(loader)}] loss: {loss.item():.4f} feature: {loss_feature.item():.4f} recon: {loss_recon.item():.4f} active: {n_active}")
        
        torch.save(lfm.state_dict(), pathconfig["lfm"][layer])
        print(f"  Saved epoch {epoch + 1}")

        # weight distribution
        weights = lfm.linear.weight.data.abs().flatten()
        total_weights = weights.numel()
        lower = 0.0
        dist_str = []
        while True:
            upper = lower + 0.1
            count = ((weights >= lower) & (weights < upper)).sum().item()
            pct = 100 * count / total_weights
            dist_str.append(f"{lower:.1f}-{upper:.1f}: {pct:.2f}%")
            if (weights >= upper).sum() == 0:
                break
            lower = upper
        print(f"  Weight dist: {' | '.join(dist_str)}")

    print(f"Done with layer {layer}\n")

if __name__ == "__main__":
    for layer in config["layers"]:
        train_layer(layer)
    print("All done!")