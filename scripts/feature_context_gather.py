import json
from collections import defaultdict
import os
from scripts.config import config, pathconfig

TOP_K = 5  # top contexts to keep per feature

os.makedirs("./data/contexts", exist_ok=True)

num_layers = config["num_layers"]


def process_file(raw_path, out_path, label):
    """Process a raw activations file into feature contexts"""
    print(f"  Processing {label}...")

    features = defaultdict(lambda: {"contexts": [], "decoding": None})

    try:
        with open(raw_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                fid = entry["feature_id"]
                features[fid]["contexts"].append({
                    "example_id": entry["example_id"],
                    "activations": entry["activations"]
                })
    except FileNotFoundError:
        print(f"    Skipping - file not found: {raw_path}")
        return 0

    # keep only top k contexts per feature by max activation
    for fid in features:
        contexts = features[fid]["contexts"]
        for ctx in contexts:
            ctx["max_act"] = max(act["activation"] for act in ctx["activations"])
        contexts.sort(key=lambda x: x["max_act"], reverse=True)
        features[fid]["contexts"] = contexts[:TOP_K]
        for ctx in features[fid]["contexts"]:
            del ctx["max_act"]

    features = {k: dict(v) for k, v in features.items()}

    with open(out_path, "w") as f:
        json.dump(features, f)

    print(f"    {len(features)} features saved")
    return len(features)


# Process all mlp positions (layers 0 to num_layers-1)
print("Processing mlp positions...")
for layer in range(num_layers):
    process_file(
        pathconfig["raw_activations"]["mlp"][layer],
        pathconfig["feature_context"]["mlp"][layer],
        f"layer {layer} mlp"
    )

# Process all att positions (layers 0 to num_layers-1)
print("\nProcessing att positions...")
for layer in range(num_layers):
    process_file(
        pathconfig["raw_activations"]["att"][layer],
        pathconfig["feature_context"]["att"][layer],
        f"layer {layer} att"
    )

print("\nDone!")
