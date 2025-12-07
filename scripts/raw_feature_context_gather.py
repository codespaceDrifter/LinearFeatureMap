"""
Gather SAE feature activations from dataset.
Uses Phi4FeatureInference which returns features directly.
Outputs separate files for mlp and att features.
"""
import json
import os
from collections import defaultdict

from datasets import load_from_disk
from phi4_mini.feature_inference import Phi4FeatureInference
from scripts.config import config, pathconfig

THRESHOLD = 0.75  # SAE activation threshold for feature firing
BATCH_SIZE = 4  # small batch for generation

os.makedirs("./data/contexts", exist_ok=True)

# load model with SAEs baked in
print("Loading Phi4 + SAEs...")
phi = Phi4FeatureInference(device=config["device"])

num_layers = config["num_layers"]

dataset = load_from_disk(pathconfig["alpaca"] + "/train")
end_idx = int(config["split"][1] * len(dataset))


def format_prompt(example):
    text = example["instruction"]
    if example["input"]:
        text += "\n" + example["input"]
    return text


# open output files (pairs only)
examples_file = open(pathconfig["example_hydrate"], "w")
files_mlp = {}
files_att = {}

for layer in range(num_layers - 1):  # 0 to 30
    files_mlp[layer] = open(pathconfig["raw_activations"]["mlp"][layer], "w")

for layer in range(1, num_layers):  # 1 to 31
    files_att[layer] = open(pathconfig["raw_activations"]["att"][layer], "w")

example_id = 0

print(f"Processing {end_idx} examples...")

for i in range(0, end_idx, BATCH_SIZE):
    batch = [dataset[j] for j in range(i, min(i + BATCH_SIZE, end_idx))]
    prompts = [format_prompt(ex) for ex in batch]

    responses, tokens, features = phi.generate(prompts, max_new_tokens=config["max_new_tokens"])

    for b in range(len(prompts)):
        examples_file.write(json.dumps({"id": example_id, "input": prompts[b], "output": responses[b]}) + "\n")

        feat = features[b]  # (64, seq_len, 12288)
        seq_len = feat.shape[1]

        # process mlp positions (layers 0 to 30)
        for layer in range(num_layers - 1):
            hook_idx = layer * 2 + 1
            z = feat[hook_idx]  # (seq_len, 12288)

            feature_fires = defaultdict(list)
            # nonzero returns a tensor of shape (num_fires, 2) the 2 is the dimension of input tensor
            # a list of lists, each inner list is a coordinate pair (token_idx, feature_idx)
            fired = (z > THRESHOLD).nonzero().tolist()

            # at the example, layer level. for each feature. notes all {token_idx, token, activation}
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

        # process att positions (layers 1 to 31)
        for layer in range(1, num_layers):
            hook_idx = layer * 2
            z = feat[hook_idx]  # (seq_len, 12288)

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
