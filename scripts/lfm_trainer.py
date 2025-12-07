"""
LFM training - can be run standalone or imported.
"""
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from interp_algo.SAE import SAE
from interp_algo.LFM import LFM
from scripts.config import config, pathconfig

BATCH_SIZE = 4096
LR = 1e-4
LAMBDA_RECON = 0.1
ACTIVE_THRESHOLD = 0.1
EPOCHS = 5


class PairedDataset(Dataset):
    def __init__(self, mlp_path, att_path, total_tokens, embed_dim):
        self.mlp = np.memmap(mlp_path, dtype=np.float32, mode="r", shape=(total_tokens, embed_dim))
        self.att = np.memmap(att_path, dtype=np.float32, mode="r", shape=(total_tokens, embed_dim))

    def __len__(self):
        return len(self.mlp)

    def __getitem__(self, idx):
        return torch.from_numpy(self.mlp[idx].copy()), torch.from_numpy(self.att[idx].copy())


def train_lfm(layer, sae_mlp, sae_att, total_tokens):
    """
    Train LFM for a layer using two SAEs.
    sae_mlp: SAE for mlp_in[layer]
    sae_att: SAE for att_in[layer+1]
    """
    print(f"Training LFM: layer {layer}")
    os.makedirs("./weights/LFM", exist_ok=True)

    dataset = PairedDataset(
        pathconfig["activations"]["mlp"][layer],
        pathconfig["activations"]["att"][layer + 1],
        total_tokens, config["embed_dim"]
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    sae_mlp.eval()
    sae_att.eval()

    lfm = LFM(config["hidden_dim"]).to(config["device"])
    optimizer = torch.optim.Adam(lfm.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for batch_idx, (mlp_in, att_in) in enumerate(loader):
            mlp_in, att_in = mlp_in.to(config["device"]), att_in.to(config["device"])

            with torch.no_grad():
                f_in = sae_mlp.encode(mlp_in)
                f_out = sae_att.encode(att_in)

            f_pred = lfm(f_in)

            active_mask = (f_in > ACTIVE_THRESHOLD) & (f_out > ACTIVE_THRESHOLD)
            if active_mask.any():
                loss_feature = ((f_out - f_pred)[active_mask]).pow(2).mean()
            else:
                loss_feature = torch.tensor(0.0, device=config["device"])

            att_in_pred = sae_att.decode(f_pred)
            loss_recon = (att_in - att_in_pred).pow(2).mean()

            loss = loss_feature + LAMBDA_RECON * loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                n_active = active_mask.sum().item()
                print(f"  [{epoch+1}/{EPOCHS}][{batch_idx}/{len(loader)}] loss: {loss.item():.4f} feat: {loss_feature.item():.4f} recon: {loss_recon.item():.4f} active: {n_active}")

        torch.save(lfm.state_dict(), pathconfig["lfm"][layer])

    # weight distribution
    weights = lfm.linear.weight.data.abs().flatten()
    n_01_02 = ((weights >= 0.1) & (weights < 0.2)).sum().item()
    n_02_plus = (weights >= 0.2).sum().item()
    print(f"  Weights 0.1-0.2: {n_01_02} | >0.2: {n_02_plus}")


# standalone execution
if __name__ == "__main__":
    metadata = np.load(pathconfig["metadata"], allow_pickle=True).item()
    total_tokens = metadata["total_tokens"]

    for layer in range(config["num_layers"] - 1):
        sae_mlp = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
        sae_mlp.load_state_dict(torch.load(pathconfig["sae"]["mlp"][layer]))

        sae_att = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
        sae_att.load_state_dict(torch.load(pathconfig["sae"]["att"][layer + 1]))

        train_lfm(layer, sae_mlp, sae_att, total_tokens)
