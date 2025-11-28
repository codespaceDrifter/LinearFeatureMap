import torch
import json
from datasets import load_from_disk
from collections import defaultdict
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from phi4_mini.inference import Phi4Inference
from SAE.SAE import SAE

LAYERS = [8, 16, 24, 31]
EMBED_DIM = 3072
HIDDEN_DIM = 12288
THRESHOLD = 0.5
DEVICE = "cuda"
MAX_NEW_TOKENS = 64

os.makedirs("./data/features", exist_ok=True)

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

# {layer_pos: {feature_id: [contexts]}}
all_features = {f"{l}_{p}": defaultdict(list) for l in LAYERS for p in ["pre", "post"]}

print(f"Processing {end_idx} examples...")

for i in range(end_idx):
    example = dataset[i]
    prompt = format_prompt(example)
    
    with torch.no_grad():
        responses, activations = phi.generate([prompt], max_new_tokens=MAX_NEW_TOKENS)
    
    output = responses[0]
    
    for layer_idx, layer in enumerate(LAYERS):
        for pos_idx, pos in enumerate(["pre", "post"]):
            acts = activations[0, :, layer_idx * 2 + pos_idx, :].to(DEVICE)

            # z shape: (seq_len, hidden_dim) e.g. (64, 12288)
            z = torch.relu(saes[f"{layer}_{pos}"].encoder(acts))
            
            # z > threshold: (seq_len, hidden_dim) e.g. (64, 12288)
            # nonzero returns coords of true values (token_idx, feature_idx)
            fired = (z > THRESHOLD).nonzero().tolist()
            
            # default dict allows append into non existing keys into a list
            feature_fires = defaultdict(list)
            for token_idx, feature_idx in fired:
                # for each feature that fired append the tuple of token index that fired and the activation value
                feature_fires[feature_idx].append((token_idx, round(z[token_idx, feature_idx].item(), 3)))
            
            for feature_idx, fires in feature_fires.items():
                all_features[f"{layer}_{pos}"][feature_idx].append({
                    "input": prompt,
                    "output": output,
                    "activations": fires
                })
    
    if i % 100 == 0:
        print(f"[{100*i/end_idx:.1f}%] {i}/{end_idx}")

print("Saving...")
with open("./data/features/all_features.json", "w") as f:
    json.dump(all_features, f)