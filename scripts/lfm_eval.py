import torch
import numpy as np
import os
from interp_algo.SAE import SAE
from interp_algo.LFM import LFM
from scripts.config import config, pathconfig

BATCH_SIZE = 4096  # eval batch size
ACTIVE_THRESHOLD = 0.1  # threshold for masked metrics

metadata = np.load(pathconfig["test_metadata"], allow_pickle=True).item()
total_tokens = metadata["total_tokens"]

def eval_layer(layer):
    print(f"\n{'='*50}")
    print(f"Evaluating LFM for layer {layer}")
    print(f"{'='*50}\n")

    # load test activations: mlp_in at layer N, att_in at layer N+1
    mlp_in = np.memmap(pathconfig["test_activations"]["mlp"][layer], dtype=np.float32, mode="r", shape=(total_tokens, config["embed_dim"]))
    att_in = np.memmap(pathconfig["test_activations"]["att"][layer + 1], dtype=np.float32, mode="r", shape=(total_tokens, config["embed_dim"]))

    # load TWO SAEs: one for input, one for output
    sae_mlp = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
    sae_mlp.load_state_dict(torch.load(pathconfig["sae"]["mlp"][layer]))
    sae_mlp.eval()

    sae_att = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
    sae_att.load_state_dict(torch.load(pathconfig["sae"]["att"][layer + 1]))
    sae_att.eval()

    lfm = LFM(config["hidden_dim"]).to(config["device"])
    lfm.load_state_dict(torch.load(pathconfig["lfm"][layer]))
    lfm.eval()
    
    # unmasked accumulators
    total_feature = 0
    total_recon = 0
    
    # masked accumulators
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
            
            # for recon, mask by token (any active feature in that token)
            token_mask = mask.any(dim=1)
            if token_mask.any():
                masked_recon_sum += ((x_out - x_out_pred)[token_mask]).pow(2).sum().item()
                masked_recon_count += token_mask.sum().item() * config["embed_dim"]
            
            num_batches += 1
    
    print("--- Unmasked (all features) ---")
    print(f"  Feature MSE: {total_feature / num_batches:.6f}")
    print(f"  Recon MSE:   {total_recon / num_batches:.6f}")
    
    print(f"\n--- Masked (f_in > {ACTIVE_THRESHOLD} AND f_out > {ACTIVE_THRESHOLD}) ---")
    if masked_feature_count > 0:
        print(f"  Feature MSE: {masked_feature_sum / masked_feature_count:.6f}")
    else:
        print(f"  Feature MSE: N/A (no active pairs)")
    if masked_recon_count > 0:
        print(f"  Recon MSE:   {masked_recon_sum / masked_recon_count:.6f}")
    else:
        print(f"  Recon MSE: N/A")
    
    print(f"\nWeight distribution:")
    weights = lfm.linear.weight.data.abs().flatten()
    total_weights = weights.numel()
    
    lower = 0.0
    while True:
        upper = lower + 0.1
        count = ((weights >= lower) & (weights < upper)).sum().item()
        pct = 100 * count / total_weights
        print(f"  {lower:.1f}-{upper:.1f}: {pct:.4f}%")
        if (weights >= upper).sum() == 0:
            break
        lower = upper

if __name__ == "__main__":
    for layer in range(config["num_layers"] - 1):
        eval_layer(layer)
    print("\nDone!")