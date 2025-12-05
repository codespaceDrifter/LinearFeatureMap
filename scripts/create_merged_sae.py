import torch
from scripts.config import config, pathconfig
from interp_algo.SAE import SAE
from interp_algo.mergedSAE import MergedSAE

layers = config["layers"]
embed_dim = config["embed_dim"]
hidden_dim = config["hidden_dim"]

print(f"Loading {len(layers)} individual SAEs...")

# load individual SAEs
saes = []
for layer in layers:
    sae = SAE(embed_dim, hidden_dim)
    sae.load_state_dict(torch.load(pathconfig["sae"][layer]))
    saes.append(sae)
    print(f"  Loaded layer {layer}")

# create merged SAE
merged = MergedSAE(len(layers), embed_dim, hidden_dim)

# copy weights
# nn.Linear weight shape: (out_features, in_features)
# we need: (in_features, out_features) for bmm
for i, sae in enumerate(saes):
    # encoder: Linear(embed_dim, hidden_dim) -> weight is (hidden_dim, embed_dim)
    # we need (embed_dim, hidden_dim) for bmm
    merged.encoder_weight.data[i] = sae.encoder.weight.data.T
    merged.encoder_bias.data[i] = sae.encoder.bias.data

    # decoder: Linear(hidden_dim, embed_dim) -> weight is (embed_dim, hidden_dim)
    # we need (hidden_dim, embed_dim) for bmm
    merged.decoder_weight.data[i] = sae.decoder.weight.data.T
    merged.decoder_bias.data[i] = sae.decoder.bias.data

print(f"Merged encoder weight shape: {merged.encoder_weight.shape}")
print(f"Merged decoder weight shape: {merged.decoder_weight.shape}")

# save
torch.save(merged.state_dict(), pathconfig["merged_sae"])
print(f"Saved to {pathconfig['merged_sae']}")
