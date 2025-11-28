import torch
import numpy as np
from datasets import load_from_disk
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from phi4_mini.inference import Phi4Inference

# config
LAYERS = [8, 16, 24, 31]
BATCH_SIZE = 4
MAX_NEW_TOKENS = 64

os.makedirs("./data/activations", exist_ok=True)

# load
dataset = load_from_disk("./data/alpaca/train")
phi = Phi4Inference(layers=LAYERS, device="cuda")

# open files in append binary mode
files = {}
for layer in LAYERS:
    files[f"layer_{layer}_pre"] = open(f"./data/activations/layer_{layer}_pre.bin", "ab")
    files[f"layer_{layer}_post"] = open(f"./data/activations/layer_{layer}_post.bin", "ab")


'''
dataset format: 
{
    "instruction": "Give three tips for staying healthy.",
    "input": "",  # often empty
    "output": "1. Eat a balanced diet..."
}
'''

def format_prompt(example):
    text = example["instruction"]
    if example["input"]:
        text += "\n" + example["input"]
    return text

# main loop
total = len(dataset)
print(f"Processing {total} examples...")

total_tokens = 0

start = 0

for i in range(start, total, BATCH_SIZE):
    batch = dataset[i:i+BATCH_SIZE]
    prompts = [format_prompt({"instruction": inst, "input": inp}) 
               for inst, inp in zip(batch["instruction"], batch["input"])]
    
    responses, activations = phi.generate(prompts, max_new_tokens=MAX_NEW_TOKENS)
    # activations: (batch, seq_len, num_layers*2, embed_dim)
    
    batch_size, seq_len, _, embed_dim = activations.shape
    total_tokens += batch_size * seq_len
    
    for layer_idx, layer in enumerate(LAYERS):
        pre_idx = layer_idx * 2
        post_idx = layer_idx * 2 + 1
        
        pre_acts = activations[:, :, pre_idx, :].reshape(-1, embed_dim).float().numpy()
        post_acts = activations[:, :, post_idx, :].reshape(-1, embed_dim).float().numpy()
        
        files[f"layer_{layer}_pre"].write(pre_acts.tobytes())
        files[f"layer_{layer}_post"].write(post_acts.tobytes())
    
    # progress + sanity check
    if i % 100 == 0:
        pct = 100 * i / total
        print(f"[{pct:.1f}%] Example {i}/{total}")
        print(f"  Input: {prompts[0][:100]}...")
        print(f"  Output: {responses[0]}")

# close files
for f in files.values():
    f.close()

# save metadata
np.save("./data/activations/metadata.npy", {
    "total_tokens": total_tokens,
    "embed_dim": 3072,
    "layers": LAYERS,
    "dtype": "float32"
})

print(f"\nDone! Total tokens: {total_tokens}")
print("Files saved to ./data/activations/")