import torch
import numpy as np
import os
from interp_algo.SAE import SAE
from scripts.config import config, pathconfig

BATCH_SIZE = 4096  # eval batch size

metadata = np.load(pathconfig["test_metadata"], allow_pickle=True).item()
total_tokens = metadata["total_tokens"]

def eval_layer(layer):
    print(f"\n{'='*50}")
    print(f"Evaluating SAE for layer {layer}")
    print(f"{'='*50}\n")

    mlp_in = np.memmap(pathconfig["test_activations"][layer]["mlp"], dtype=np.float32, mode="r", shape=(total_tokens, config["embed_dim"]))
    att_in = np.memmap(pathconfig["test_activations"][layer]["att"], dtype=np.float32, mode="r", shape=(total_tokens, config["embed_dim"]))

    weight_path = pathconfig["sae"][layer]
    if not os.path.exists(weight_path):
        print(f"Skipping - no weights found")
        return

    sae = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
    sae.load_state_dict(torch.load(weight_path))
    sae.eval()
    
    # eval on mlp_in
    print("--- mlp_in ---")
    total_recon, total_norm, total_active, total_mid, total_strong, count = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for i in range(0, total_tokens, BATCH_SIZE):
            x = torch.from_numpy(mlp_in[i:i+BATCH_SIZE].copy()).to(config["device"])
            x_hat, z = sae(x)
            total_recon += (x - x_hat).pow(2).sum().item()
            total_norm += x.pow(2).sum().item()
            total_active += (z > 0).float().sum().item()
            total_mid += (z > 0.5).float().sum().item()
            total_strong += (z > 1).float().sum().item()
            count += x.shape[0]
    
    print(f"  R²: {1 - total_recon / total_norm:.4f}")
    print(f"  Active (>0): {total_active / count:.1f}")
    print(f"  Mid (>0.5): {total_mid / count:.1f}")
    print(f"  Strong (>1): {total_strong / count:.1f}")
    
    # eval on att_in
    print(f"--- att_in (layer {layer+1}) ---")
    total_recon, total_norm, total_active, total_mid, total_strong, count = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for i in range(0, total_tokens, BATCH_SIZE):
            x = torch.from_numpy(att_in[i:i+BATCH_SIZE].copy()).to(config["device"])
            x_hat, z = sae(x)
            total_recon += (x - x_hat).pow(2).sum().item()
            total_norm += x.pow(2).sum().item()
            total_active += (z > 0).float().sum().item()
            total_mid += (z > 0.5).float().sum().item()
            total_strong += (z > 1).float().sum().item()
            count += x.shape[0]
    
    print(f"  R²: {1 - total_recon / total_norm:.4f}")
    print(f"  Active (>0): {total_active / count:.1f}")
    print(f"  Mid (>0.5): {total_mid / count:.1f}")
    print(f"  Strong (>1): {total_strong / count:.1f}")

if __name__ == "__main__":
    for layer in config["layers"]:
        eval_layer(layer)
    print("\nDone!")