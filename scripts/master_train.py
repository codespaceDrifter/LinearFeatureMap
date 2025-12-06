"""
Master training script - trains all SAEs and LFMs layer by layer.
Imports modular functions from other scripts.
"""
from datasets import load_from_disk
from phi4_mini.inference import Phi4Inference
from scripts.config import config, pathconfig
from scripts.activation_gather import gather_pair, delete_pair
from scripts.sae_trainer import train_sae
from scripts.lfm_trainer import train_lfm

# load model and dataset once
print("Loading Phi4...")
phi = Phi4Inference(device=config["device"])

print("Loading dataset...")
dataset = load_from_disk(pathconfig["alpaca"] + "/train")
total = len(dataset)
start = int(config["split"][0] * total)
end = int(config["split"][1] * total)

num_layers = config["num_layers"]

print(f"\nTraining {num_layers - 1} layer pairs...")
print(f"  SAEs: {(num_layers - 1) * 2} total (mlp[0-{num_layers-2}] + att[1-{num_layers-1}])")
print(f"  LFMs: {num_layers - 1} total\n")

for layer in range(num_layers - 1):
    print(f"\n{'='*60}")
    print(f"Layer {layer}: mlp_in[{layer}] -> att_in[{layer+1}]")
    print(f"{'='*60}")

    # 1. gather activations
    total_tokens = gather_pair(phi, dataset, layer, start, end)

    # 2. train SAEs
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

    # 3. train LFM
    train_lfm(layer, sae_mlp, sae_att, total_tokens)

    # 4. delete activations
    delete_pair(layer)

    print(f"\nLayer {layer} complete!")

print("\n" + "="*60)
print("All done!")
print("="*60)
print(f"\nSAEs: ./weights/SAE/")
print(f"LFMs: ./weights/LFM/")
print(f"\nNext: run interpretation pipeline")
