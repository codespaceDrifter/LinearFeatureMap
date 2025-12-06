"""
Evaluate SAEs on test set.
"""
import torch
import numpy as np
import os
from interp_algo.SAE import SAE
from scripts.config import config, pathconfig

BATCH_SIZE = 4096

num_layers = config["num_layers"]


def eval_sae(kind, layer, total_tokens):
    """Evaluate a single SAE."""
    print(f"\n--- SAE {kind}[{layer}] ---")

    # load test activations
    act_path = pathconfig["test_activations"][kind][layer]
    if not os.path.exists(act_path):
        print(f"  Skipping - no test activations")
        return

    activations = np.memmap(act_path, dtype=np.float32, mode="r", shape=(total_tokens, config["embed_dim"]))

    # load SAE
    weight_path = pathconfig["sae"][kind][layer]
    if not os.path.exists(weight_path):
        print(f"  Skipping - no weights")
        return

    sae = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
    sae.load_state_dict(torch.load(weight_path, weights_only=True))
    sae.eval()

    total_recon, total_norm, total_active, total_mid, total_strong, count = 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for i in range(0, total_tokens, BATCH_SIZE):
            x = torch.from_numpy(activations[i:i+BATCH_SIZE].copy()).to(config["device"])
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


# standalone execution
if __name__ == "__main__":
    metadata = np.load(pathconfig["test_metadata"], allow_pickle=True).item()
    total_tokens = metadata["total_tokens"]

    print("Evaluating mlp SAEs...")
    for layer in range(num_layers - 1):
        eval_sae("mlp", layer, total_tokens)

    print("\nEvaluating att SAEs...")
    for layer in range(1, num_layers):
        eval_sae("att", layer, total_tokens)

    print("\nDone!")
