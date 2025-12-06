"""
Evaluate LFMs on test set.
"""
import torch
import numpy as np
import os
from interp_algo.SAE import SAE
from interp_algo.LFM import LFM
from scripts.config import config, pathconfig

BATCH_SIZE = 4096
ACTIVE_THRESHOLD = 0.1


def eval_lfm(layer, total_tokens):
    """Evaluate LFM for a single layer."""
    print(f"\n--- LFM layer {layer} ---")

    # load test activations
    mlp_path = pathconfig["test_activations"]["mlp"][layer]
    att_path = pathconfig["test_activations"]["att"][layer + 1]

    if not os.path.exists(mlp_path) or not os.path.exists(att_path):
        print(f"  Skipping - no test activations")
        return

    mlp_in = np.memmap(mlp_path, dtype=np.float32, mode="r", shape=(total_tokens, config["embed_dim"]))
    att_in = np.memmap(att_path, dtype=np.float32, mode="r", shape=(total_tokens, config["embed_dim"]))

    # load SAEs
    sae_mlp_path = pathconfig["sae"]["mlp"][layer]
    sae_att_path = pathconfig["sae"]["att"][layer + 1]
    lfm_path = pathconfig["lfm"][layer]

    if not all(os.path.exists(p) for p in [sae_mlp_path, sae_att_path, lfm_path]):
        print(f"  Skipping - missing weights")
        return

    sae_mlp = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
    sae_mlp.load_state_dict(torch.load(sae_mlp_path, weights_only=True))
    sae_mlp.eval()

    sae_att = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
    sae_att.load_state_dict(torch.load(sae_att_path, weights_only=True))
    sae_att.eval()

    lfm = LFM(config["hidden_dim"]).to(config["device"])
    lfm.load_state_dict(torch.load(lfm_path, weights_only=True))
    lfm.eval()

    # accumulators
    total_feature = 0
    total_recon = 0
    masked_feature_sum = 0
    masked_feature_count = 0
    masked_recon_sum = 0
    masked_recon_count = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx in range(total_tokens // BATCH_SIZE):
            start = batch_idx * BATCH_SIZE
            end = start + BATCH_SIZE

            x_in = torch.from_numpy(mlp_in[start:end].copy()).to(config["device"])
            x_out = torch.from_numpy(att_in[start:end].copy()).to(config["device"])

            f_in = sae_mlp.encode(x_in)
            f_out = sae_att.encode(x_out)

            f_pred = lfm(f_in)
            x_out_pred = sae_att.decode(f_pred)

            # unmasked
            total_feature += (f_out - f_pred).pow(2).mean().item()
            total_recon += (x_out - x_out_pred).pow(2).mean().item()

            # masked (both in and out active)
            mask = (f_in > ACTIVE_THRESHOLD) & (f_out > ACTIVE_THRESHOLD)
            if mask.any():
                masked_feature_sum += ((f_out - f_pred)[mask]).pow(2).sum().item()
                masked_feature_count += mask.sum().item()

            token_mask = mask.any(dim=1)
            if token_mask.any():
                masked_recon_sum += ((x_out - x_out_pred)[token_mask]).pow(2).sum().item()
                masked_recon_count += token_mask.sum().item() * config["embed_dim"]

            num_batches += 1

    print(f"  Unmasked - Feature MSE: {total_feature / num_batches:.6f}, Recon MSE: {total_recon / num_batches:.6f}")
    if masked_feature_count > 0:
        print(f"  Masked   - Feature MSE: {masked_feature_sum / masked_feature_count:.6f}", end="")
        if masked_recon_count > 0:
            print(f", Recon MSE: {masked_recon_sum / masked_recon_count:.6f}")
        else:
            print()

    # weight distribution
    weights = lfm.linear.weight.data.abs().flatten()
    n_01_02 = ((weights >= 0.1) & (weights < 0.2)).sum().item()
    n_02_plus = (weights >= 0.2).sum().item()
    print(f"  Weights 0.1-0.2: {n_01_02} | >0.2: {n_02_plus}")


# standalone execution
if __name__ == "__main__":
    metadata = np.load(pathconfig["test_metadata"], allow_pickle=True).item()
    total_tokens = metadata["total_tokens"]

    for layer in range(config["num_layers"] - 1):
        eval_lfm(layer, total_tokens)

    print("\nDone!")
