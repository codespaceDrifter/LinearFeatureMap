import torch
import json
from datasets import load_from_disk
from collections import defaultdict
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from phi4_mini.inference import Phi4Inference
from interp_algo.SAE import SAE

LAYERS = [8, 16, 24, 31]
EMBED_DIM = 3072
HIDDEN_DIM = 12288
THRESHOLD = 0.5
DEVICE = "cuda"
MAX_NEW_TOKENS = 64
BATCH_SIZE = 4

os.makedirs("./data/features_jsonl", exist_ok=True)

phi = Phi4Inference(layers=LAYERS, device=DEVICE)

saes = {}
for layer in LAYERS:
    for pos in ["pre", "post"]:
        sae = SAE(EMBED_DIM, HIDDEN_DIM).to(DEVICE)
        sae.load_state_dict(torch.load(f"./weights/SAE/layer_{layer}_{pos}_sae.pt"))
        sae.eval()
        saes[f"{layer}_{pos}"] = sae

dataset = load_from_disk("./data/alpaca/train")
end_idx = int(0.75 * len(dataset))

def format_prompt(example):
    text = example["instruction"]
    if example["input"]:
        text += "\n" + example["input"]
    return text

examples_file = open("./data/examples.jsonl", "a")
files = {}
for layer in LAYERS:
    for pos in ["pre", "post"]:
        files[f"{layer}_{pos}"] = open(f"./data/features_jsonl/layer_{layer}_{pos}.jsonl", "a")

example_id = 0

print(f"Processing {end_idx} examples...")

for i in range(0, end_idx, BATCH_SIZE):
    batch = [dataset[j] for j in range(i, min(i + BATCH_SIZE, end_idx))]
    prompts = [format_prompt(ex) for ex in batch]
    
    with torch.no_grad():
        responses, tokens, activations = phi.generate(prompts, max_new_tokens=MAX_NEW_TOKENS)
    
    for b in range(len(prompts)):
        examples_file.write(json.dumps({"id": example_id, "input": prompts[b], "output": responses[b]}) + "\n")
        
        for layer_idx, layer in enumerate(LAYERS):
            for pos_idx, pos in enumerate(["pre", "post"]):
                acts = activations[b, :, layer_idx * 2 + pos_idx, :].float().to(DEVICE)
                z = torch.relu(saes[f"{layer}_{pos}"].encoder(acts))
                
                fired = (z > THRESHOLD).nonzero().tolist()
                
                feature_fires = defaultdict(list)
                for token_idx, feature_idx in fired:
                    feature_fires[feature_idx].append((token_idx, tokens[b][token_idx], round(z[token_idx, feature_idx].item(), 3)))
                
                for feature_idx, fires in feature_fires.items():
                    files[f"{layer}_{pos}"].write(json.dumps({"feature_id": feature_idx, "example_id": example_id, "activations": fires}) + "\n")
        
        example_id += 1
    
    if i % 100 < BATCH_SIZE:
        print(f"[{100*i/end_idx:.1f}%] {i}/{end_idx}")

examples_file.close()
for f in files.values():
    f.close()

print("Done!")