import json
from collections import defaultdict
import os
from scripts.config import config, pathconfig

TOP_K = 5  # top contexts to keep per feature

os.makedirs("./data/contexts", exist_ok=True)

for layer in config["layers"]:
    print(f"Processing layer {layer}...")

    features = defaultdict(lambda: {"contexts": [], "decoding": None})

    with open(pathconfig["raw_activations"][layer], "r") as f:
        for line in f:
            entry = json.loads(line)
            fid = entry["feature_id"]
            features[fid]["contexts"].append({
                "example_id": entry["example_id"],
                "activations": entry["activations"]
            })

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

    with open(pathconfig["feature_context"][layer], "w") as f:
        json.dump(features, f)

    print(f"  {len(features)} features saved (top {TOP_K} contexts each)")

print("Done!")
