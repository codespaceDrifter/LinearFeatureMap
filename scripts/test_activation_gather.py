import torch
import numpy as np
from datasets import load_from_disk
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from phi4_mini.inference import Phi4Inference

LAYERS = [8, 16, 24, 31]
BATCH_SIZE = 4
MAX_NEW_TOKENS = 64
SAVE_DIR = "./data/test/activations"
SPLIT = (0.75, 1.0)

os.makedirs(SAVE_DIR, exist_ok=True)
dataset = load_from_disk("./data/alpaca/train")
phi = Phi4Inference(layers=LAYERS, device="cuda")

files = {}
for layer in LAYERS:
    files[f"layer_{layer}_pre"] = open(f"{SAVE_DIR}/layer_{layer}_pre.bin", "ab")
    files[f"layer_{layer}_post"] = open(f"{SAVE_DIR}/layer_{layer}_post.bin", "ab")

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
    batch_size, seq_len, _, embed_dim = activations.shape
    total_tokens += batch_size * seq_len
    
    for layer_idx, layer in enumerate(LAYERS):
        pre_acts = activations[:, :, layer_idx * 2, :].reshape(-1, embed_dim).float().numpy()
        post_acts = activations[:, :, layer_idx * 2 + 1, :].reshape(-1, embed_dim).float().numpy()
        files[f"layer_{layer}_pre"].write(pre_acts.tobytes())
        files[f"layer_{layer}_post"].write(post_acts.tobytes())
    
    if (i-start) % 100 == 0:
        print(f"[{100*i/total:.1f}%] {i}/{total} | {prompts[0][:60]}... -> {responses[0][:60]}...")

for f in files.values():
    f.close()

np.save(f"{SAVE_DIR}/metadata.npy", {"total_tokens": total_tokens, "embed_dim": 3072, "layers": LAYERS, "dtype": "float32"})
print(f"Done! {total_tokens} tokens saved to {SAVE_DIR}")