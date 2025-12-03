import torch
import numpy as np
from datasets import load_from_disk
import os
from phi4_mini.inference import Phi4Inference

LAYERS = [8, 16, 24]
BATCH_SIZE = 4
MAX_NEW_TOKENS = 64
SAVE_DIR = "./data/test/activations"
SPLIT = (0.75, 1.0)

os.makedirs(SAVE_DIR, exist_ok=True)
dataset = load_from_disk("./data/alpaca/train")
phi = Phi4Inference(device="cuda")

files = {}
for layer in LAYERS:
    files[f"layer_{layer}_mlp_in"] = open(f"{SAVE_DIR}/layer_{layer}_mlp_in.bin", "ab")
    files[f"layer_{layer+1}_att_in"] = open(f"{SAVE_DIR}/layer_{layer+1}_att_in.bin", "ab")

def format_prompt(example):
    text = example["instruction"]
    if example["input"]:
        text += "\n" + example["input"]
    return text

total = len(dataset)
start = int(SPLIT[0] * total)
end = int(SPLIT[1] * total)
total_tokens = 0

print(f"Processing examples {start} to {end}...")

for i in range(start, end, BATCH_SIZE):
    batch = dataset[i:i+BATCH_SIZE]
    prompts = [format_prompt({"instruction": inst, "input": inp}) 
               for inst, inp in zip(batch["instruction"], batch["input"])]
    
    responses, _, activations = phi.generate(prompts, max_new_tokens=MAX_NEW_TOKENS)
    # activations: [Tensor(64, seq_len_b, 3072), ...] length B
    
    for act in activations:
        # act: Tensor(64, seq_len, 3072)
        seq_len = act.shape[1]
        total_tokens += seq_len
        
        for layer in LAYERS:
            mlp_in_idx = layer * 2 + 1
            att_in_idx = (layer + 1) * 2
            
            mlp_in = act[mlp_in_idx, :, :].float().numpy()  # (seq_len, 3072)
            att_in = act[att_in_idx, :, :].float().numpy()  # (seq_len, 3072)
            
            files[f"layer_{layer}_mlp_in"].write(mlp_in.tobytes())
            files[f"layer_{layer+1}_att_in"].write(att_in.tobytes())
    
    if (i - start) % 100 == 0:
        print(f"[{100*i/total:.1f}%] {i}/{total} | {prompts[0]}... -> {responses[0]}...")

for f in files.values():
    f.close()

np.save(f"{SAVE_DIR}/metadata.npy", {"total_tokens": total_tokens, "embed_dim": 3072, "layers": LAYERS, "dtype": "float32"})
print(f"Done! {total_tokens} tokens saved to {SAVE_DIR}")