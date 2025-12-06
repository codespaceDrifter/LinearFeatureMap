"""
Master training script - trains all SAEs and LFMs in layer batches.
Gathers activations for multiple layers at once to avoid repeated inference.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_from_disk
from phi4_mini.inference import Phi4Inference
from scripts.config import config, pathconfig
from scripts.activation_gather import gather_layer_batch, delete_layer_batch
from scripts.sae_trainer import train_sae
from scripts.lfm_trainer import train_lfm

LAYER_BATCH_SIZE = 8  # gather 8 layers at once (~450GB disk, adjust if needed)

# load model and dataset once
print("Loading Phi4...")
phi = Phi4Inference(device=config["device"])

print("Loading dataset...")
dataset = load_from_disk(pathconfig["alpaca"] + "/train")
total = len(dataset)
start = int(config["split"][0] * total)
end = int(config["split"][1] * total)

num_layers = config["num_layers"]

print(f"\nTraining {num_layers - 1} layer pairs in batches of {LAYER_BATCH_SIZE}...")
print(f"  SAEs: {(num_layers - 1) * 2} total (mlp[0-{num_layers-2}] + att[1-{num_layers-1}])")
print(f"  LFMs: {num_layers - 1} total\n")

# process in layer batches
for batch_start in range(0, num_layers - 1, LAYER_BATCH_SIZE):
    batch_end = min(batch_start + LAYER_BATCH_SIZE, num_layers - 1)
    layers = list(range(batch_start, batch_end))

    print(f"\n{'#'*60}")
    print(f"LAYER BATCH: {batch_start}-{batch_end-1} ({len(layers)} layers)")
    print(f"{'#'*60}")

    # 1. gather activations for all layers in batch (one pass through dataset)
    total_tokens = gather_layer_batch(phi, dataset, layers, start, end)

    # 2. train SAEs and LFMs for each layer in batch
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}: mlp_in[{layer}] -> att_in[{layer+1}]")
        print(f"{'='*60}")

        # train SAEs
        sae_mlp = train_sae(
            pathconfig["activations"]["mlp"][layer],
            pathconfig["sae"]["mlp"][layer],
            total_tokens
        )
        sae_att = train_sae(
            pathconfig["activations"]["att"][layer + 1],
            pathconfig["sae"]["att"][layer + 1],
            total_tokens
        )

        # train LFM
        train_lfm(layer, sae_mlp, sae_att, total_tokens)

        print(f"\nLayer {layer} complete!")

    # 3. delete all activations for this batch
    delete_layer_batch(layers)

    print(f"\nBatch {batch_start}-{batch_end-1} complete!")

print("\n" + "="*60)
print("All done!")
print("="*60)
print(f"\nSAEs: ./weights/SAE/")
print(f"LFMs: ./weights/LFM/")
print(f"\nNext: run interpretation pipeline")
