import json
import copy

# load examples
examples = {}
with open("./data/examples.jsonl", "r") as f:
    for line in f:
        ex = json.loads(line)
        examples[ex["id"]] = {"input": ex["input"], "output": ex["output"]}

with open("./data/features/layer_8_pre.json", "r") as f:
    features = json.load(f)

for fid, fdata in features.items():
    hydrated = copy.deepcopy(fdata)
    for ctx in hydrated["contexts"]:
        ex = examples.get(ctx["example_id"])
        if ex:
            ctx["input"] = ex["input"]
            ctx["output"] = ex["output"]
            del ctx["example_id"]
    
    print(f"\n{'='*60}")
    print(f"Feature {fid}")
    print(f"{'='*60}")
    print(json.dumps(hydrated, indent=2))
    
    input("\n[Enter]")
