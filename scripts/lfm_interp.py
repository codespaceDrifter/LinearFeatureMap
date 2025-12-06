import torch
import json
import os
from scripts.config import config, pathconfig
from interp_algo.LFM import LFM

WEIGHT_THRESHOLD = 0.15

os.makedirs("./data/labels", exist_ok=True)

for layer in range(config["num_layers"] - 1):
    print(f"Processing layer {layer}...")

    # load TWO interpretation files:
    # - mlp interpretations for input features (from sae_mlp[layer])
    # - att interpretations for output features (from sae_att[layer+1])
    try:
        with open(pathconfig["interpretations"]["mlp"][layer], "r") as f:
            interps_in = json.load(f)
    except FileNotFoundError:
        print(f"  Skipping - no mlp interpretations for layer {layer}")
        continue

    try:
        with open(pathconfig["interpretations"]["att"][layer + 1], "r") as f:
            interps_out = json.load(f)
    except FileNotFoundError:
        print(f"  Skipping - no att interpretations for layer {layer + 1}")
        continue

    def get_label_in(fid):
        fid_str = str(fid)
        if fid_str in interps_in:
            return interps_in[fid_str].get("interpretation", "")
        return ""

    def get_label_out(fid):
        fid_str = str(fid)
        if fid_str in interps_out:
            return interps_out[fid_str].get("interpretation", "")
        return ""

    lfm = LFM(config["hidden_dim"])
    lfm.load_state_dict(torch.load(pathconfig["lfm"][layer]))

    # W[i, j] = weight mapping feature_in j -> feature_out i
    weights = lfm.linear.weight.data  # (hidden_dim, hidden_dim)

    # find all weights > threshold
    # structure: {feature_in: {label, outputs: [{id, label, weight}, ...]}}
    result = {}

    # find all (i, j) where weight > threshold
    mask = weights > WEIGHT_THRESHOLD
    indices = mask.nonzero().tolist()  # list of [i, j]

    for i, j in indices:
        # skip self-connections (same feature index, though different SAEs)
        if i == j:
            continue

        # skip if input or output not labeled
        j_label = get_label_in(j)
        i_label = get_label_out(i)
        if not j_label or not i_label:
            continue

        j_str = str(j)
        if j_str not in result:
            result[j_str] = {
                "label": j_label,
                "outputs": []
            }
        result[j_str]["outputs"].append({
            "id": i,
            "label": i_label,
            "weight": round(weights[i, j].item(), 4)
        })

    # sort each feature_in's outputs by weight descending
    for j_str in result:
        result[j_str]["outputs"].sort(key=lambda x: x["weight"], reverse=True)

    out_path = pathconfig["lfm_labels"][layer]
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    total_connections = sum(len(v["outputs"]) for v in result.values())
    print(f"  {len(result)} input features with {total_connections} connections (>{WEIGHT_THRESHOLD})")

print("Done!")
