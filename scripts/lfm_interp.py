import torch
import json
import os
from scripts.config import config, pathconfig
from interp_algo.LFM import LFM

WEIGHT_THRESHOLD = 0.15

os.makedirs("./data/labels", exist_ok=True)

for layer in config["layers"]:
    print(f"Processing layer {layer}...")

    # load interpretations
    with open(pathconfig["interpretations"][layer], "r") as f:
        interps = json.load(f)

    def get_label(fid):
        fid_str = str(fid)
        if fid_str in interps:
            return interps[fid_str].get("interpretation", "")
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
        j_str = str(j)
        if j_str not in result:
            result[j_str] = {
                "label": get_label(j),
                "outputs": []
            }
        result[j_str]["outputs"].append({
            "id": i,
            "label": get_label(i),
            "weight": round(weights[i, j].item(), 4)
        })

    # sort each feature_in's outputs by weight descending
    for j_str in result:
        result[j_str]["outputs"].sort(key=lambda x: x["weight"], reverse=True)

    out_path = f"./data/labels/lfm_{layer}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    total_connections = sum(len(v["outputs"]) for v in result.values())
    print(f"  {len(result)} input features with {total_connections} connections (>{WEIGHT_THRESHOLD})")

print("Done!")
