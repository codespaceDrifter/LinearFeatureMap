"""
Gather test activations for evaluation.
Processes all layer pairs: mlp[0-30] + att[1-31]
"""
import torch
import numpy as np
from datasets import load_from_disk
import os
from phi4_mini.inference import Phi4Inference
from scripts.config import config, pathconfig

BATCH_SIZE = 4

os.makedirs("./data/test/activations", exist_ok=True)

num_layers = config["num_layers"]

def gather_test_activations():
    """Gather test set activations for all layer pairs."""
    dataset = load_from_disk(pathconfig["alpaca"] + "/train")
    phi = Phi4Inference(device=config["device"])

    # open files for all pairs
    files_mlp = {}
    files_att = {}

    for layer in range(num_layers - 1):  # 0 to 30
        files_mlp[layer] = open(pathconfig["test_activations"]["mlp"][layer], "ab")

    for layer in range(1, num_layers):  # 1 to 31
        files_att[layer] = open(pathconfig["test_activations"]["att"][layer], "ab")

    def format_prompt(example):
        text = example["instruction"]
        if example["input"]:
            text += "\n" + example["input"]
        return text

    total = len(dataset)
    start = int(config["split"][1] * total)  # test starts where train ends
    end = total
    total_tokens = 0

    print(f"Processing test examples {start} to {end}...")

    for i in range(start, end, BATCH_SIZE):
        batch = dataset[i:i+BATCH_SIZE]
        prompts = [format_prompt({"instruction": inst, "input": inp})
                   for inst, inp in zip(batch["instruction"], batch["input"])]

        responses, _, activations = phi.generate(prompts, max_new_tokens=config["max_new_tokens"])

        for act in activations:
            seq_len = act.shape[1]
            total_tokens += seq_len

            # mlp positions (layers 0 to 30)
            for layer in range(num_layers - 1):
                mlp_in_idx = layer * 2 + 1
                mlp_in = act[mlp_in_idx, :, :].float().numpy()
                files_mlp[layer].write(mlp_in.tobytes())

            # att positions (layers 1 to 31)
            for layer in range(1, num_layers):
                att_in_idx = layer * 2
                att_in = act[att_in_idx, :, :].float().numpy()
                files_att[layer].write(att_in.tobytes())

        if (i - start) % 100 < BATCH_SIZE:
            print(f"[{100*(i-start)/(end-start):.1f}%] {i-start}/{end-start}")

    for f in files_mlp.values():
        f.close()
    for f in files_att.values():
        f.close()

    np.save(pathconfig["test_metadata"], {
        "total_tokens": total_tokens,
        "embed_dim": config["embed_dim"],
        "num_layers": num_layers,
        "dtype": "float32"
    })

    print(f"Done! {total_tokens} tokens saved")
    return total_tokens


if __name__ == "__main__":
    gather_test_activations()
