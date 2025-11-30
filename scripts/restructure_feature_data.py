import json
from collections import defaultdict
import os

LAYERS = [8, 16, 24]
TOP_K = 20

os.makedirs("./data/features", exist_ok=True)

for layer in LAYERS:
    for pos in ["pre", "post"]:
        print(f"Processing layer_{layer}_{pos}...")
        
        features = defaultdict(lambda: {"contexts": [], "decoding": None, "interpretation": None})
        
        with open(f"./data/features_jsonl/layer_{layer}_{pos}.jsonl", "r") as f:
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
                ctx["max_act"] = max(act[2] for act in ctx["activations"])
            contexts.sort(key=lambda x: x["max_act"], reverse=True)
            features[fid]["contexts"] = contexts[:TOP_K]
            for ctx in features[fid]["contexts"]:
                del ctx["max_act"]
        
        features = {k: dict(v) for k, v in features.items()}
        
        with open(f"./data/features/layer_{layer}_{pos}.json", "w") as f:
            json.dump(features, f)
        
        print(f"  {len(features)} features saved (top {TOP_K} contexts each)")

print("Done!")