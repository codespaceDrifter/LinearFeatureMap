"""
Adds decoder similarity info to feature context files.
For each feature, finds the top-K tokens most similar to its decoder vector.
Updates the existing context files in-place (adds "decoding" field).
"""
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from interp_algo.SAE import SAE
from scripts.config import config, pathconfig

TOP_K = 4  # top decoded tokens to keep

# load model for embeddings and tokenizer
print("Loading model for embeddings...")
model = AutoModelForCausalLM.from_pretrained(
    pathconfig["model"], dtype=torch.bfloat16, trust_remote_code=False
).to(config["device"])
tokenizer = AutoTokenizer.from_pretrained(pathconfig["model"], trust_remote_code=False)

token_embeddings = model.model.embed_tokens.weight.detach().float()  # (vocab_size, 3072)
embed_norm = token_embeddings / token_embeddings.norm(dim=1, keepdim=True)

# free model memory, keep only embeddings
del model
torch.cuda.empty_cache()

num_layers = config["num_layers"]


def process_layer(kind, layer):
    """Add decoding info to a single layer's feature context file."""
    context_path = pathconfig["feature_context"][kind][layer]
    sae_path = pathconfig["sae"][kind][layer]

    with open(context_path, "r") as f:
        features = json.load(f)

    # load SAE
    sae = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
    sae.load_state_dict(torch.load(sae_path, weights_only=True))
    sae.eval()

    decoder = sae.decoder.weight.detach().float()  # (embed_dim, hidden_dim)

    for fid in features:
        feature_vector = decoder[:, int(fid)]  # (3072,)
        feature_norm = feature_vector / feature_vector.norm()

        similarities = embed_norm @ feature_norm  # (vocab_size,)
        top_k = similarities.topk(TOP_K)

        top_tokens = [tokenizer.decode([i]) for i in top_k.indices.tolist()]

        features[fid]["decoding"] = top_tokens

    # write back to same file
    with open(context_path, "w") as f:
        json.dump(features, f)

    # free SAE memory
    del sae
    torch.cuda.empty_cache()

    return len(features)


# Process mlp layers (0 to 30)
print("Processing mlp layers...")
for layer in range(num_layers - 1):
    count = process_layer("mlp", layer)
    if count > 0:
        print(f"  layer {layer} mlp: {count} features updated")

# Process att layers (1 to 31)
print("\nProcessing att layers...")
for layer in range(1, num_layers):
    count = process_layer("att", layer)
    if count > 0:
        print(f"  layer {layer} att: {count} features updated")

print("\nDone!")
