import torch
import json
from datasets import load_from_disk
from collections import defaultdict
import os
from phi4_mini.inference import Phi4Inference
from interp_algo.SAE import SAE

LAYERS = [8, 16, 24]
EMBED_DIM = 3072
HIDDEN_DIM = EMBED_DIM * 4
THRESHOLD = 0.5
DEVICE = "cuda"
MAX_NEW_TOKENS = 64
BATCH_SIZE = 4

os.makedirs("./data/features_jsonl", exist_ok=True)

phi = Phi4Inference(device=DEVICE)

saes = {}
for layer in LAYERS:
    sae = SAE(EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    sae.load_state_dict(torch.load(f"./weights/SAE/layer_{layer}_sae.pt"))
    sae.eval()
    saes[layer] = sae

dataset = load_from_disk("./data/alpaca/train")
end_idx = int(0.75 * len(dataset))

def format_prompt(example):
    text = example["instruction"]
    if example["input"]:
        text += "\n" + example["input"]
    return text

examples_file = open("./data/examples.jsonl", "w")
files = {}
for layer in LAYERS:
    files[layer] = open(f"./data/features_jsonl/layer_{layer}.jsonl", "w")

example_id = 0

print(f"Processing {end_idx} examples...")

for i in range(0, end_idx, BATCH_SIZE):
    batch = [dataset[j] for j in range(i, min(i + BATCH_SIZE, end_idx))]
    prompts = [format_prompt(ex) for ex in batch]
    
    with torch.no_grad():
        responses, tokens, activations = phi.generate(prompts, max_new_tokens=MAX_NEW_TOKENS)
    # activations: [Tensor(64, seq_len_b, 3072), ...] length B
    
    for b in range(len(prompts)):
        examples_file.write(json.dumps({"id": example_id, "input": prompts[b], "output": responses[b]}) + "\n")
        
        act = activations[b]  # Tensor(64, seq_len, 3072)
        seq_len = act.shape[1]
        
        for layer in LAYERS:
            sae = saes[layer]
            
            mlp_in_idx = layer * 2 + 1
            att_in_idx = (layer + 1) * 2
            
            mlp_in_acts = act[mlp_in_idx, :, :].float().to(DEVICE)  # (seq_len, 3072)
            att_in_acts = act[att_in_idx, :, :].float().to(DEVICE)  # (seq_len, 3072)
            
            # encode both through same SAE
            z_mlp = sae.encode(mlp_in_acts)  # (seq_len, hidden_dim)
            z_att = sae.encode(att_in_acts)  # (seq_len, hidden_dim)
            
            feature_fires = defaultdict(list)
            
            # collect from mlp_in
            fired = (z_mlp > THRESHOLD).nonzero().tolist()  # [[token_idx, feature_idx], ...]
            for token_idx, feature_idx in fired:
                feature_fires[feature_idx].append({
                    "pos": "mlp_in",
                    "token_idx": token_idx,
                    "token": tokens[b][token_idx],
                    "activation": round(z_mlp[token_idx, feature_idx].item(), 3)
                })
            
            # collect from att_in
            fired = (z_att > THRESHOLD).nonzero().tolist()
            for token_idx, feature_idx in fired:
                feature_fires[feature_idx].append({
                    "pos": "att_in",
                    "token_idx": token_idx,
                    "token": tokens[b][token_idx],
                    "activation": round(z_att[token_idx, feature_idx].item(), 3)
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