"""
Master training script for all layers.
Processes pairs of positions (mlp_in at layer N, att_in at layer N+1) together,
trains both SAEs and the LFM, then deletes activations.
This keeps disk usage to ~2 positions at a time (~100GB) instead of 1.8TB total.
"""
import torch
import torch.nn as nn
import numpy as np
import os
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from phi4_mini.inference import Phi4Inference
from interp_algo.SAE import SAE
from interp_algo.LFM import LFM
from scripts.config import config, pathconfig

# SAE training params
BATCH_SIZE_GATHER = 4
BATCH_SIZE_SAE = 16384
SAE_LR = 1e-4
SAE_LAMBDA_SPARSE = 1
SAE_EPOCHS = 20

# LFM training params
BATCH_SIZE_LFM = 4096
LFM_LR = 1e-4
LFM_LAMBDA_RECON = 0.1
LFM_ACTIVE_THRESHOLD = 0.1
LFM_EPOCHS = 5

os.makedirs("./weights/SAE", exist_ok=True)
os.makedirs("./weights/LFM", exist_ok=True)
os.makedirs("./data/activations", exist_ok=True)

# load model once
print("Loading Phi4...")
phi = Phi4Inference(device=config["device"])

# load dataset
ds = load_from_disk(pathconfig["alpaca"] + "/train")
total = len(ds)
start = int(config["split"][0] * total)
end = int(config["split"][1] * total)


def format_prompt(example):
    text = example["instruction"]
    if example["input"]:
        text += "\n" + example["input"]
    return text


def gather_pair(layer):
    """Gather mlp_in at layer N and att_in at layer N+1 together (one inference pass)"""
    print(f"\n--- Gathering pair: layer {layer} mlp_in + layer {layer+1} att_in ---")

    mlp_hook_idx = layer * 2 + 1
    att_hook_idx = (layer + 1) * 2

    mlp_path = pathconfig["activations"]["mlp"][layer]
    att_path = pathconfig["activations"]["att"][layer + 1]

    f_mlp = open(mlp_path, "wb")
    f_att = open(att_path, "wb")
    total_tokens = 0

    for i in range(start, end, BATCH_SIZE_GATHER):
        batch = ds[i:i+BATCH_SIZE_GATHER]
        prompts = [format_prompt({"instruction": inst, "input": inp})
                   for inst, inp in zip(batch["instruction"], batch["input"])]

        with torch.no_grad():
            _, _, activations = phi.generate(prompts, max_new_tokens=config["max_new_tokens"])

        for act in activations:
            seq_len = act.shape[1]
            total_tokens += seq_len

            mlp_data = act[mlp_hook_idx, :, :].float().numpy()
            att_data = act[att_hook_idx, :, :].float().numpy()

            f_mlp.write(mlp_data.tobytes())
            f_att.write(att_data.tobytes())

        if i % 100 < BATCH_SIZE_GATHER:
            print(f"  [{100*(i-start)/(end-start):.1f}%] {i}/{end}")

    f_mlp.close()
    f_att.close()
    print(f"  {total_tokens} tokens saved")
    return total_tokens


def train_sae(path, save_path, total_tokens):
    """Train SAE on activation data"""
    data = np.memmap(path, dtype=np.float32, mode="r", shape=(total_tokens, config["embed_dim"]))

    class SimpleDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return torch.from_numpy(self.data[idx].copy())

    loader = DataLoader(SimpleDataset(data), batch_size=BATCH_SIZE_SAE, shuffle=True, num_workers=4)

    sae = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
    optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_LR)

    for epoch in range(SAE_EPOCHS):
        for batch_idx, x in enumerate(loader):
            x = x.to(config["device"])
            x_hat, z = sae(x)

            recon_loss = (x - x_hat).pow(2).mean()
            sparse_loss = z.mean()
            loss = recon_loss + SAE_LAMBDA_SPARSE * sparse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                any_active = (z > 0).float().mean().item() * config["hidden_dim"]
                r_squared = 1 - (x - x_hat).pow(2).sum() / x.pow(2).sum()
                print(f"    [{epoch+1}/{SAE_EPOCHS}][{batch_idx}/{len(loader)}] loss: {loss.item():.4f} R²: {r_squared.item():.4f} active: {any_active:.0f}")

        torch.save(sae.state_dict(), save_path)

    return sae


def train_lfm(layer, sae_mlp, sae_att, total_tokens):
    """Train LFM using two different SAEs"""
    print(f"\n--- Training LFM for layer {layer} ---")

    mlp_path = pathconfig["activations"]["mlp"][layer]
    att_path = pathconfig["activations"]["att"][layer + 1]

    mlp_data = np.memmap(mlp_path, dtype=np.float32, mode="r", shape=(total_tokens, config["embed_dim"]))
    att_data = np.memmap(att_path, dtype=np.float32, mode="r", shape=(total_tokens, config["embed_dim"]))

    class PairedDataset(Dataset):
        def __init__(self, mlp, att):
            self.mlp = mlp
            self.att = att
        def __len__(self):
            return len(self.mlp)
        def __getitem__(self, idx):
            return torch.from_numpy(self.mlp[idx].copy()), torch.from_numpy(self.att[idx].copy())

    loader = DataLoader(PairedDataset(mlp_data, att_data), batch_size=BATCH_SIZE_LFM, shuffle=True, num_workers=4, pin_memory=True)

    sae_mlp.eval()
    sae_att.eval()

    lfm = LFM(config["hidden_dim"]).to(config["device"])
    optimizer = torch.optim.Adam(lfm.parameters(), lr=LFM_LR)

    for epoch in range(LFM_EPOCHS):
        for batch_idx, (mlp_in, att_in) in enumerate(loader):
            mlp_in, att_in = mlp_in.to(config["device"]), att_in.to(config["device"])

            with torch.no_grad():
                f_in = sae_mlp.encode(mlp_in)
                f_out = sae_att.encode(att_in)

            f_pred = lfm(f_in)

            # masked feature loss
            active_mask = (f_in > LFM_ACTIVE_THRESHOLD) & (f_out > LFM_ACTIVE_THRESHOLD)
            if active_mask.any():
                loss_feature = ((f_out - f_pred)[active_mask]).pow(2).mean()
            else:
                loss_feature = torch.tensor(0.0, device=config["device"])

            # recon loss
            att_in_pred = sae_att.decode(f_pred)
            loss_recon = (att_in - att_in_pred).pow(2).mean()

            loss = loss_feature + LFM_LAMBDA_RECON * loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                n_active = active_mask.sum().item()
                print(f"    [{epoch+1}/{LFM_EPOCHS}][{batch_idx}/{len(loader)}] loss: {loss.item():.4f} feat: {loss_feature.item():.4f} recon: {loss_recon.item():.4f} active: {n_active}")

        torch.save(lfm.state_dict(), pathconfig["lfm"][layer])

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


def delete_pair(layer):
    """Delete activation files for a pair"""
    for path in [pathconfig["activations"]["mlp"][layer], pathconfig["activations"]["att"][layer + 1]]:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Deleted {path}")


def process_layer(layer):
    """Full pipeline for one LFM: gather pair -> train SAEs -> train LFM -> delete"""
    print(f"\n{'='*60}")
    print(f"Layer {layer}: mlp_in[{layer}] -> att_in[{layer+1}]")
    print(f"{'='*60}")

    # 1. Gather activations (one inference pass for both)
    total_tokens = gather_pair(layer)

    # 2. Train SAE for mlp_in
    print(f"\n--- Training SAE for layer {layer} mlp_in ---")
    sae_mlp = train_sae(
        pathconfig["activations"]["mlp"][layer],
        pathconfig["sae"]["mlp"][layer],
        total_tokens
    )

    # 3. Train SAE for att_in
    print(f"\n--- Training SAE for layer {layer+1} att_in ---")
    sae_att = train_sae(
        pathconfig["activations"]["att"][layer + 1],
        pathconfig["sae"]["att"][layer + 1],
        total_tokens
    )

    # 4. Train LFM
    train_lfm(layer, sae_mlp, sae_att, total_tokens)

    # 5. Delete activations
    delete_pair(layer)

    print(f"\nLayer {layer} complete!")


def gather_single(layer, kind):
    """Gather activations for a single position"""
    print(f"\n--- Gathering layer {layer} {kind}_in ---")

    if kind == "mlp":
        hook_idx = layer * 2 + 1
    else:  # att
        hook_idx = layer * 2

    out_path = pathconfig["activations"][kind][layer]
    f = open(out_path, "wb")
    total_tokens = 0

    for i in range(start, end, BATCH_SIZE_GATHER):
        batch = ds[i:i+BATCH_SIZE_GATHER]
        prompts = [format_prompt({"instruction": inst, "input": inp})
                   for inst, inp in zip(batch["instruction"], batch["input"])]

        with torch.no_grad():
            _, _, activations = phi.generate(prompts, max_new_tokens=config["max_new_tokens"])

        for act in activations:
            seq_len = act.shape[1]
            total_tokens += seq_len
            data = act[hook_idx, :, :].float().numpy()
            f.write(data.tobytes())

        if i % 100 < BATCH_SIZE_GATHER:
            print(f"  [{100*(i-start)/(end-start):.1f}%] {i}/{end}")

    f.close()
    print(f"  {total_tokens} tokens saved")
    return total_tokens


def train_single_sae(layer, kind):
    """Train SAE for a single position (no LFM)"""
    print(f"\n{'='*60}")
    print(f"Training SAE only: layer {layer} {kind}_in")
    print(f"{'='*60}")

    total_tokens = gather_single(layer, kind)

    print(f"\n--- Training SAE ---")
    train_sae(
        pathconfig["activations"][kind][layer],
        pathconfig["sae"][kind][layer],
        total_tokens
    )

    # delete activations
    path = pathconfig["activations"][kind][layer]
    if os.path.exists(path):
        os.remove(path)
        print(f"  Deleted {path}")


if __name__ == "__main__":
    num_layers = config["num_layers"]

    # 1. First: train att_in[0] SAE alone (no LFM - it's the first input)
    train_single_sae(0, "att")

    # 2. Middle: train pairs + LFMs
    # For layer N: train mlp_in[N] SAE, att_in[N+1] SAE, and LFM mapping between them
    for layer in range(num_layers - 1):
        process_layer(layer)

    # 3. Last: train mlp_in[31] SAE alone (no LFM - no att_in[32] target)
    train_single_sae(num_layers - 1, "mlp")

    print("\n" + "="*60)
    print("All SAEs and LFMs trained!")
    print("="*60)
    print(f"\nSAEs: 64 total (32 mlp + 32 att)")
    print(f"LFMs: 31 total (mlp[N] -> att[N+1] for N=0-30)")
    print(f"\nWeights saved in ./weights/SAE/ and ./weights/LFM/")
    print(f"Next: run interpretation pipeline")
