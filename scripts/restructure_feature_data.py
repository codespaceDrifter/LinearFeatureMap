import json
from collections import defaultdict
import os

LAYERS = [8, 16, 24]

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
        
        features = {k: dict(v) for k, v in features.items()}
        
        with open(f"./data/features/layer_{layer}_{pos}.json", "w") as f:
            json.dump(features, f)
        
        print(f"  {len(features)} features saved")

print("Done!")