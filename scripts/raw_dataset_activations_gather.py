import torch
import json
from datasets import load_from_disk
from collections import defaultdict
import os
from phi4_mini.inference import Phi4Inference
from interp_algo.mergedSAE import MergedSAE
from scripts.config import config, pathconfig

THRESHOLD = 0.75  # SAE activation threshold for feature firing
BATCH_SIZE = 4  # small batch for generation

os.makedirs("./data/contexts", exist_ok=True)

# load model
phi = Phi4Inference(device=config["device"])

# load merged SAE - one forward pass for all layers
print("Loading MergedSAE...")
layers = config["layers"]
num_layers = len(layers)
merged_sae = MergedSAE(num_layers, config["embed_dim"], config["hidden_dim"]).to(config["device"])
merged_sae.load_state_dict(torch.load(pathconfig["merged_sae"]))
merged_sae.eval()
print("Loaded MergedSAE")

# precompute layer indices
mlp_indices = [layer * 2 + 1 for layer in layers]
att_indices = [(layer + 1) * 2 for layer in layers]

dataset = load_from_disk(pathconfig["alpaca"] + "/train")
end_idx = int(config["split"][1] * len(dataset))

def format_prompt(example):
    text = example["instruction"]
    if example["input"]:
        text += "\n" + example["input"]
    return text

examples_file = open(pathconfig["example_hydrate"], "w")
files = {}
for layer in layers:
    files[layer] = open(pathconfig["raw_activations"][layer], "w")

example_id = 0

print(f"Processing {end_idx} examples...")

for i in range(0, end_idx, BATCH_SIZE):
    batch = [dataset[j] for j in range(i, min(i + BATCH_SIZE, end_idx))]
    prompts = [format_prompt(ex) for ex in batch]

    with torch.no_grad():
        responses, tokens, activations = phi.generate(prompts, max_new_tokens=config["max_new_tokens"])
        # activations: [Tensor(64, seq_len_b, 3072), ...] length B

        for b in range(len(prompts)):
            examples_file.write(json.dumps({"id": example_id, "input": prompts[b], "output": responses[b]}) + "\n")

            act = activations[b]  # (64, seq_len, 3072)
            seq_len = act.shape[1]

            # stack all layer activations: (seq, num_layers, embed_dim)
            mlp_in_stack = torch.stack([act[idx, :, :] for idx in mlp_indices], dim=1).float().to(config["device"])
            att_in_stack = torch.stack([act[idx, :, :] for idx in att_indices], dim=1).float().to(config["device"])

            # encode separately to save memory
            z_mlp = merged_sae.encode(mlp_in_stack)  # (seq, num_layers, hidden_dim)
            z_att = merged_sae.encode(att_in_stack)  # (seq, num_layers, hidden_dim)

            # process each layer
            for layer_idx, layer in enumerate(layers):
                z_mlp_layer = z_mlp[:, layer_idx, :]  # (seq, hidden_dim)
                z_att_layer = z_att[:, layer_idx, :]  # (seq, hidden_dim)

                feature_fires = defaultdict(list)

                # collect from mlp_in
                fired = (z_mlp_layer > THRESHOLD).nonzero().tolist()
                for token_idx, feature_idx in fired:
                    feature_fires[feature_idx].append({
                        "pos": "mlp_in",
                        "token_idx": token_idx,
                        "token": tokens[b][token_idx],
                        "activation": round(z_mlp_layer[token_idx, feature_idx].item(), 3)
                    })

                # collect from att_in
                fired = (z_att_layer > THRESHOLD).nonzero().tolist()
                for token_idx, feature_idx in fired:
                    feature_fires[feature_idx].append({
                        "pos": "att_in",
                        "token_idx": token_idx,
                        "token": tokens[b][token_idx],
                        "activation": round(z_att_layer[token_idx, feature_idx].item(), 3)
                    })

                for feature_idx, fires in feature_fires.items():
                    files[layer].write(json.dumps({
                        "feature_id": feature_idx,
                        "example_id": example_id,
                        "activations": fires
                    }) + "\n")

            example_id += 1

    if i % 100 < BATCH_SIZE:
        print(f"[{100*i/end_idx:.1f}%] {i}/{end_idx}")

examples_file.close()
for f in files.values():
    f.close()

print("Done!")
