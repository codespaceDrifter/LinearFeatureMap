"""
Activation gathering - can be run standalone or imported.
"""
import torch
import numpy as np
from datasets import load_from_disk
import os
from scripts.config import config, pathconfig

BATCH_SIZE = 4


def format_prompt(example):
    text = example["instruction"]
    if example["input"]:
        text += "\n" + example["input"]
    return text


def gather_pair(phi, dataset, layer, start, end):
    """
    Gather activations for a layer pair: mlp_in[layer] and att_in[layer+1].
    Returns total_tokens.
    """
    print(f"Gathering layer {layer}: mlp_in[{layer}] + att_in[{layer+1}]")

    os.makedirs("./data/activations", exist_ok=True)

    mlp_path = pathconfig["activations"]["mlp"][layer]
    att_path = pathconfig["activations"]["att"][layer + 1]

    f_mlp = open(mlp_path, "wb")
    f_att = open(att_path, "wb")
    total_tokens = 0

    mlp_idx = layer * 2 + 1
    att_idx = (layer + 1) * 2

    for i in range(start, end, BATCH_SIZE):
        batch = dataset[i:i+BATCH_SIZE]
        prompts = [format_prompt({"instruction": inst, "input": inp})
                   for inst, inp in zip(batch["instruction"], batch["input"])]

        with torch.no_grad():
            _, _, activations = phi.generate(prompts, max_new_tokens=config["max_new_tokens"])

        for act in activations:
            seq_len = act.shape[1]
            total_tokens += seq_len

            f_mlp.write(act[mlp_idx, :, :].float().numpy().tobytes())
            f_att.write(act[att_idx, :, :].float().numpy().tobytes())

        if i % 100 < BATCH_SIZE:
            print(f"  [{100*(i-start)/(end-start):.1f}%] {i}/{end}")

    f_mlp.close()
    f_att.close()
    print(f"  {total_tokens} tokens saved")
    return total_tokens


def delete_pair(layer):
    """Delete activation files for a layer pair."""
    for path in [pathconfig["activations"]["mlp"][layer], pathconfig["activations"]["att"][layer + 1]]:
        if os.path.exists(path):
            os.remove(path)


# standalone execution
if __name__ == "__main__":
    from phi4_mini.inference import Phi4Inference

    phi = Phi4Inference(device=config["device"])
    dataset = load_from_disk(pathconfig["alpaca"] + "/train")

    total = len(dataset)
    start = int(config["split"][0] * total)
    end = int(config["split"][1] * total)

    for layer in range(config["num_layers"] - 1):
        gather_pair(phi, dataset, layer, start, end)