import torch
import numpy as np
from datasets import load_from_disk
import os
from phi4_mini.inference import Phi4Inference
from scripts.config import config, pathconfig

BATCH_SIZE = 4  # small batch for generation

SAVE_DIR = "./data/activations"
os.makedirs(SAVE_DIR, exist_ok=True)
dataset = load_from_disk(pathconfig["alpaca"] + "/train")
phi = Phi4Inference(device=config["device"])

# because we are studying the MLP so we want the post attention norm of this layer and the input norm of the next layer
# for layer N MLP: input is mlp_in (index N*2+1), output is att_in of layer N+1 (index (N+1)*2)
files = {}
for layer in config["layers"]:
    files[f"layer_{layer}_mlp_in"] = open(pathconfig["activations"][layer]["mlp"], "ab")
    files[f"layer_{layer+1}_att_in"] = open(pathconfig["activations"][layer]["att"], "ab")

def format_prompt(example):
    text = example["instruction"]
    if example["input"]:
        text += "\n" + example["input"]
    return text

total = len(dataset)
start = int(config["split"][0] * total)
end = int(config["split"][1] * total)
total_tokens = 0

print(f"Processing examples {start} to {end}...")

for i in range(start, end, BATCH_SIZE):
    batch = dataset[i:i+BATCH_SIZE]
    prompts = [format_prompt({"instruction": inst, "input": inp}) 
               for inst, inp in zip(batch["instruction"], batch["input"])]
    
    responses, _, activations = phi.generate(prompts, max_new_tokens=config["max_new_tokens"])
    # activations: [Tensor(64, seq_len_b, 3072), ...] length B
    
    for act in activations:
        # act: Tensor(64, seq_len, 3072)
        seq_len = act.shape[1]
        total_tokens += seq_len
        
        for layer in config["layers"]:
            mlp_in_idx = layer * 2 + 1  # mlp_in for layer N
            att_in_idx = (layer + 1) * 2  # att_in for layer N+1
            
            mlp_in = act[mlp_in_idx, :, :].float().numpy()  # (seq_len, 3072)
            att_in = act[att_in_idx, :, :].float().numpy()  # (seq_len, 3072)
            
            files[f"layer_{layer}_mlp_in"].write(mlp_in.tobytes())
            files[f"layer_{layer+1}_att_in"].write(att_in.tobytes())
    
    if i % 100 == 0:
        print(f"[{100*i/total:.1f}%] {i}/{total} | {prompts[0]}... -> {responses[0]}...")

for f in files.values():
    f.close()

np.save(pathconfig["metadata"], {"total_tokens": total_tokens, "embed_dim": config["embed_dim"], "layers": config["layers"], "dtype": "float32"})
print(f"Done! {total_tokens} tokens saved to {SAVE_DIR}")