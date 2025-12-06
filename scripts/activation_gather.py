"""
Activation gathering - can be run standalone or imported.
Gathers in batches of layers for efficiency.
"""
import glob
import os

import torch
import numpy as np
from datasets import load_from_disk
from scripts.config import config, pathconfig

BATCH_SIZE = 4  # inference batch size


def format_prompt(example):
    text = example["instruction"]
    if example["input"]:
        text += "\n" + example["input"]
    return text


def gather_layer_batch(phi, dataset, layers, start, end):
    """
    Gather activations for a batch of layers.

    For each layer N in layers:
      - saves mlp_in[N] (if N <= 30)
      - saves att_in[N+1] (if N+1 >= 1, i.e., always for valid N)

    This runs inference once per data batch and saves to all layer files,
    instead of running inference separately per layer.

    Args:
        phi: Phi4Inference model
        dataset: loaded dataset
        layers: list of layer indices to gather (e.g., [0,1,2,3,4,5,6,7])
        start: dataset start index
        end: dataset end index

    Returns:
        total_tokens
    """
    num_layers = config["num_layers"]

    # validate layers
    layers = [l for l in layers if 0 <= l < num_layers - 1]  # only 0-30 valid
    if not layers:
        return 0

    print(f"Gathering layers {layers[0]}-{layers[-1]}: {len(layers)} layer pairs")

    # clear any leftover .bin files from previous runs
    for f in glob.glob("./data/activations/*.bin"):
        os.remove(f)
        print(f"  Deleted leftover: {f}")

    os.makedirs("./data/activations", exist_ok=True)

    # open all files
    files_mlp = {}
    files_att = {}

    for layer in layers:
        files_mlp[layer] = open(pathconfig["activations"]["mlp"][layer], "wb")
        files_att[layer + 1] = open(pathconfig["activations"]["att"][layer + 1], "wb")

    total_tokens = 0

    for i in range(start, end, BATCH_SIZE):
        batch = dataset[i:i+BATCH_SIZE]
        prompts = [format_prompt({"instruction": inst, "input": inp})
                   for inst, inp in zip(batch["instruction"], batch["input"])]

        with torch.no_grad():
            _, _, activations = phi.generate(prompts, max_new_tokens=config["max_new_tokens"])

        for act in activations:
            seq_len = act.shape[1]
            total_tokens += seq_len

            for layer in layers:
                mlp_idx = layer * 2 + 1
                att_idx = (layer + 1) * 2

                files_mlp[layer].write(act[mlp_idx, :, :].float().numpy().tobytes())
                files_att[layer + 1].write(act[att_idx, :, :].float().numpy().tobytes())

        if i % 100 < BATCH_SIZE:
            print(f"  [{100*(i-start)/(end-start):.1f}%] {i}/{end}")

    for f in files_mlp.values():
        f.close()
    for f in files_att.values():
        f.close()

    print(f"  {total_tokens} tokens saved for {len(layers)} layer pairs")
    return total_tokens


def delete_layer_batch(layers):
    """Delete activation files for a batch of layers."""
    for layer in layers:
        for path in [pathconfig["activations"]["mlp"][layer], pathconfig["activations"]["att"][layer + 1]]:
            if os.path.exists(path):
                os.remove(path)
    print(f"Deleted activations for layers {layers[0]}-{layers[-1]}")


# standalone execution
if __name__ == "__main__":
    from phi4_mini.inference import Phi4Inference

    LAYER_BATCH_SIZE = 8

    phi = Phi4Inference(device=config["device"])
    dataset = load_from_disk(pathconfig["alpaca"] + "/train")

    total = len(dataset)
    start = int(config["split"][0] * total)
    end = int(config["split"][1] * total)

    num_layers = config["num_layers"]

    for batch_start in range(0, num_layers - 1, LAYER_BATCH_SIZE):
        batch_end = min(batch_start + LAYER_BATCH_SIZE, num_layers - 1)
        layers = list(range(batch_start, batch_end))
        gather_layer_batch(phi, dataset, layers, start, end)
