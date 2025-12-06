"""
Master evaluation script - evaluates all SAEs and LFMs in layer batches.
Same structure as master_train but uses test split and runs eval instead of train.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from datasets import load_from_disk
from phi4_mini.inference import Phi4Inference
from scripts.config import config, pathconfig
from scripts.test_activation_gather import gather_test_layer_batch, delete_test_layer_batch
from scripts.sae_eval import eval_sae
from scripts.lfm_eval import eval_lfm

LAYER_BATCH_SIZE = 8  # gather 8 layers at once

# load model and dataset once
print("Loading Phi4...")
phi = Phi4Inference(device=config["device"])

print("Loading dataset...")
dataset = load_from_disk(pathconfig["alpaca"] + "/train")
total = len(dataset)
start = int(config["split"][1] * total)  # test starts where train ends
end = total

num_layers = config["num_layers"]

print(f"\nEvaluating {num_layers - 1} layer pairs in batches of {LAYER_BATCH_SIZE}...")
print(f"  SAEs: {(num_layers - 1) * 2} total (mlp[0-{num_layers-2}] + att[1-{num_layers-1}])")
print(f"  LFMs: {num_layers - 1} total\n")

# process in layer batches
for batch_start in range(0, num_layers - 1, LAYER_BATCH_SIZE):
    batch_end = min(batch_start + LAYER_BATCH_SIZE, num_layers - 1)
    layers = list(range(batch_start, batch_end))

    print(f"\n{'#'*60}")
    print(f"LAYER BATCH: {batch_start}-{batch_end-1} ({len(layers)} layers)")
    print(f"{'#'*60}")

    # 1. gather test activations for all layers in batch
    total_tokens = gather_test_layer_batch(phi, dataset, layers, start, end)

    # save metadata for this batch (eval functions need it)
    np.save(pathconfig["test_metadata"], {
        "total_tokens": total_tokens,
        "embed_dim": config["embed_dim"],
        "num_layers": num_layers,
        "dtype": "float32"
    })

    # 2. eval SAEs and LFMs for each layer in batch
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}: mlp_in[{layer}] -> att_in[{layer+1}]")
        print(f"{'='*60}")

        # eval SAEs
        eval_sae("mlp", layer, total_tokens)
        eval_sae("att", layer + 1, total_tokens)

        # eval LFM
        eval_lfm(layer, total_tokens)

        print(f"\nLayer {layer} eval complete!")

    # 3. delete test activations for this batch
    delete_test_layer_batch(layers)

    print(f"\nBatch {batch_start}-{batch_end-1} complete!")

print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)
