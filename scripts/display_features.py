import json

# load examples for lookup
examples = {}
with open("./data/examples.jsonl", "r") as f:
    for line in f:
        ex = json.loads(line)
        examples[ex["id"]] = {"input": ex["input"], "output": ex["output"]}

layer = input("Layer (8/16/24): ")
pos = input("Position (pre/post): ")

with open(f"./data/features/layer_{layer}_{pos}.json", "r") as f:
    features = json.load(f)

print(f"\nLoaded {len(features)} features\n")

for fid, fdata in features.items():
    print("=" * 60)
    print(f"FEATURE {fid} | Layer {layer} {pos}")
    print("=" * 60)
    
    # decoding
    decoding = fdata.get("decoding", {})
    if decoding:
        tokens = decoding.get("tokens", [])
        scores = decoding.get("scores", [])
        print(f"\nTop decoded tokens:")
        for t, s in zip(tokens[:10], scores[:10]):
            print(f"  {t:20} ({s})")
    
    # interpretation if exists
    interp = fdata.get("interpretation")
    if interp:
        print(f"\nInterpretation: {interp}")
    
    # contexts
    print(f"\nContexts ({len(fdata['contexts'])} total):")
    for i, ctx in enumerate(fdata["contexts"][:5]):
        ex = examples.get(ctx["example_id"], {"input": "?", "output": "?"})
        print(f"\n  --- Example {i+1} ---")
        print(f"  Input:  {ex['input'][:150]}...")
        print(f"  Output: {ex['output'][:150]}...")
        print(f"  Activations:")
        for tok_idx, tok, val in ctx["activations"][:5]:
            print(f"    [{tok_idx}] '{tok}' = {val}")
    
    print("\n")
    cmd = input("Enter to continue, 'q' to quit: ")
    if cmd.lower() == 'q':
        break