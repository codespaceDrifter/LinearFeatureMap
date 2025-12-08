import json
import anthropic
import os
from scripts.config import config, pathconfig

client = anthropic.Anthropic()

num_layers = config["num_layers"]

# load examples for hydration
examples = {}
with open(pathconfig["example_hydrate"], "r") as f:
    for line in f:
        ex = json.loads(line)
        examples[ex["id"]] = {"input": ex["input"], "output": ex["output"]}

ZERO_SHOT_PROMPT = """
Label features for mechanistic interpretability of Sparse Autoencoder.
You'll see:
1. Contexts where this feature activated (top 4 highest activating examples). Each context includes:
   - input: the input question/prompt
   - output: the model's response
   - tokens: tokens where this feature activated strongly
2. Top decoded tokens: top 4 most similar tokens to the feature's decoder vector

Your job:
one shot interp the feature label without thinking or filler or "ok let me start now" or reasoning steps. your output should START AND END WITH THE CONCISE FEATURE LABEL.
make sure the feature interpretation you draft actually explains many or most of the contexts pretty well. 
top decoded could provide clues but the middle layers might not be actually embedding space so maybe they won't make sense. 
also the context can provide clues but maybe it just numerically fits that it fires without a strong semantic meaning. 
make sure it is CONCISE AND CLEAR within 10 words or so.
Output only your label, nothing else. because i will directly display and read your output as the labels
"""


os.makedirs("./data/features", exist_ok=True)

all_batch_ids = []
all_requests = []


def process_features(kind, layer):
    """Process features for a position (mlp or att)"""
    with open(pathconfig["feature_context"][kind][layer], "r") as f:
        features = json.load(f)

    requests = []
    for fid, fdata in features.items():
        # hydrate contexts
        hydrated_contexts = []
        for ctx in fdata["contexts"]:
            ex = examples.get(ctx["example_id"])
            if ex is None:
                continue
            hydrated_contexts.append({
                "input": ex["input"],
                "output": ex["output"],
                "tokens": ctx["tokens"]
            })

        user_msg = f"Feature {fid} | layer {layer} {kind}_in\n\n"
        user_msg += f"Contexts:\n{json.dumps(hydrated_contexts, indent=2)}\n\n"
        user_msg += f"Top decoded: {fdata['decoding']}"

        # custom_id format: kind_layer_fid (e.g., "mlp_5_1234")
        requests.append({
            "custom_id": f"{kind}_{layer}_{fid}",
            "params": {
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 50,
                "thinking": {"type": "disabled"},
                "system": ZERO_SHOT_PROMPT,
                "messages": [{"role": "user", "content": user_msg}]
            }
        })

    print(f"  {kind}[{layer}]: {len(requests)} features")
    return requests


# Process mlp positions (layers 0 to 30)
print("Processing mlp positions...")
for layer in range(num_layers - 1):
    all_requests.extend(process_features("mlp", layer))

# Process att positions (layers 1 to 31)
print("\nProcessing att positions...")
for layer in range(1, num_layers):
    all_requests.extend(process_features("att", layer))

print(f"\nTotal: {len(all_requests)} requests")

# Submit in chunks
CHUNK_SIZE = 4000
for i in range(0, len(all_requests), CHUNK_SIZE):
    chunk = all_requests[i:i+CHUNK_SIZE]
    print(f"  Submitting chunk {i//CHUNK_SIZE + 1}: {len(chunk)} requests...")
    batch = client.messages.batches.create(requests=chunk)
    all_batch_ids.append(batch.id)
    print(f"    Batch ID: {batch.id}")

with open(pathconfig["batch_ids"], "w") as f:
    f.write("\n".join(all_batch_ids))

print(f"\nAll batches submitted! {len(all_batch_ids)} total")
