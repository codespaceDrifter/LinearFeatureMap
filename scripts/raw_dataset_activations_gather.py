"""
Gather SAE feature activations from dataset.
Now uses separate SAEs for each position (mlp and att).
Outputs separate files for mlp and att features.
"""
import torch
import json
from datasets import load_from_disk
from collections import defaultdict
import os
from phi4_mini.inference import Phi4Inference
from interp_algo.SAE import SAE
from scripts.config import config, pathconfig

THRESHOLD = 0.75  # SAE activation threshold for feature firing
BATCH_SIZE = 4  # small batch for generation

os.makedirs("./data/contexts", exist_ok=True)

# load model
phi = Phi4Inference(device=config["device"])

num_layers = config["num_layers"]

# load all SAEs (64 total: 32 mlp + 32 att)
print("Loading SAEs...")
saes_mlp = {}
saes_att = {}

for layer in range(num_layers):
    sae = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
    sae.load_state_dict(torch.load(pathconfig["sae"]["mlp"][layer]))
    sae.eval()
    saes_mlp[layer] = sae
    print(f"  Loaded sae_mlp[{layer}]")

for layer in range(num_layers):
    sae = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
    sae.load_state_dict(torch.load(pathconfig["sae"]["att"][layer]))
    sae.eval()
    saes_att[layer] = sae
    print(f"  Loaded sae_att[{layer}]")

print(f"Loaded {len(saes_mlp)} mlp SAEs and {len(saes_att)} att SAEs")

dataset = load_from_disk(pathconfig["alpaca"] + "/train")
end_idx = int(config["split"][1] * len(dataset))

def format_prompt(example):
    text = example["instruction"]
    if example["input"]:
        text += "\n" + example["input"]
    return text

# open output files
examples_file = open(pathconfig["example_hydrate"], "w")
files_mlp = {}
files_att = {}

for layer in range(num_layers):
    files_mlp[layer] = open(pathconfig["raw_activations"]["mlp"][layer], "w")

for layer in range(num_layers):
    files_att[layer] = open(pathconfig["raw_activations"]["att"][layer], "w")

example_id = 0

print(f"Processing {end_idx} examples...")

for i in range(0, end_idx, BATCH_SIZE):
    batch = [dataset[j] for j in range(i, min(i + BATCH_SIZE, end_idx))]
    prompts = [format_prompt(ex) for ex in batch]

    with torch.no_grad():
        responses, tokens, activations = phi.generate(prompts, max_new_tokens=config["max_new_tokens"])

        for b in range(len(prompts)):
            examples_file.write(json.dumps({"id": example_id, "input": prompts[b], "output": responses[b]}) + "\n")

            act = activations[b]  # (64, seq_len, 3072)
            seq_len = act.shape[1]

            # process all mlp positions (layers 0 to num_layers-1)
            for layer in range(num_layers):
                hook_idx = layer * 2 + 1
                x = act[hook_idx, :, :].float().to(config["device"])  # (seq, embed_dim)
                z = saes_mlp[layer].encode(x)  # (seq, hidden_dim)

                feature_fires = defaultdict(list)
                fired = (z > THRESHOLD).nonzero().tolist()
                for token_idx, feature_idx in fired:
                    feature_fires[feature_idx].append({
                        "token_idx": token_idx,
                        "token": tokens[b][token_idx],
                        "activation": round(z[token_idx, feature_idx].item(), 3)
                    })

                for feature_idx, fires in feature_fires.items():
                    files_mlp[layer].write(json.dumps({
                        "feature_id": feature_idx,
                        "example_id": example_id,
                        "activations": fires
                    }) + "\n")

            # process all att positions (layers 0 to num_layers-1)
            for layer in range(num_layers):
                hook_idx = layer * 2
                x = act[hook_idx, :, :].float().to(config["device"])  # (seq, embed_dim)
                z = saes_att[layer].encode(x)  # (seq, hidden_dim)

                feature_fires = defaultdict(list)
                fired = (z > THRESHOLD).nonzero().tolist()
                for token_idx, feature_idx in fired:
                    feature_fires[feature_idx].append({
                        "token_idx": token_idx,
                        "token": tokens[b][token_idx],
                        "activation": round(z[token_idx, feature_idx].item(), 3)
                    })

                for feature_idx, fires in feature_fires.items():
                    files_att[layer].write(json.dumps({
                        "feature_id": feature_idx,
                        "example_id": example_id,
                        "activations": fires
                    }) + "\n")

            example_id += 1

    if i % 100 < BATCH_SIZE:
        print(f"[{100*i/end_idx:.1f}%] {i}/{end_idx}")

examples_file.close()
for f in files_mlp.values():
    f.close()
for f in files_att.values():
    f.close()

print("Done!")
