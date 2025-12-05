import torch
import json
from scripts.config import config, pathconfig
from interp_algo.LFM import LFM

WEIGHT_THRESHOLD = 0.1

for layer in config["layers"]:
    print(f"Processing layer {layer}...")

    lfm = LFM(config["hidden_dim"])
    lfm.load_state_dict(torch.load(pathconfig["lfm"][layer]))

    # W[i, j] = weight mapping feature_in j -> feature_out i
    weights = lfm.linear.weight.data  # (hidden_dim, hidden_dim)

    # find all weights > threshold
    # structure: {feature_in: [{feature_out, weight}, ...]}
    result = {}

    # find all (i, j) where weight > threshold
    mask = weights > WEIGHT_THRESHOLD
    indices = mask.nonzero().tolist()  # list of [i, j]

    for i, j in indices:
        j_str = str(j)
        if j_str not in result:
            result[j_str] = []
        result[j_str].append({
            "feature_out": i,
            "weight": round(weights[i, j].item(), 4)
        })

    # sort each feature_in's outputs by weight descending
    for j_str in result:
        result[j_str].sort(key=lambda x: x["weight"], reverse=True)

    out_path = f"./weights/LFM/lfm_{layer}_interp.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    total_connections = sum(len(v) for v in result.values())
    print(f"  {len(result)} input features with {total_connections} connections (>{WEIGHT_THRESHOLD})")

print("Done!")
