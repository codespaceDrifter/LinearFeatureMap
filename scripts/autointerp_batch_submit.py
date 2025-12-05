import json
import copy
import anthropic
import os
from scripts.config import config, pathconfig

client = anthropic.Anthropic()

# load examples for hydration
examples = {}
with open(pathconfig["example_hydrate"], "r") as f:
    for line in f:
        ex = json.loads(line)
        examples[ex["id"]] = {"input": ex["input"], "output": ex["output"]}


ZERO_SHOT_PROMPT = """
Hi claudy! You will help me label features for mechanistic interpretability of Sparse Autoencoder.
You'll see:
1. Contexts where this feature activated (we trained on model outputs in the alpaca dataset) Context includes:
Activations. which are the tokens in the answer that led this feature to fire. formated as [index] , 'token' , activation value
the input question in the dataset
the output answer in the model wrote
2. Top decoded tokens: top 5 dot product of decoder weights in SAE for that feature with all token embeddings (both normalized)
formatted as first the list of top decoded tokens and then their corresponding scores

Your job:
one shot interp the feature label without thinking or filler or "ok let me start now" or reasoning steps. your output should START AND END WITH THE CONCISE FEATURE LABEL.
make sure the feature interpretation you draft actually explains many or most of the contexts pretty well. top decoded could provide clues but the middle layers might not be actually
embedding space so maybe they won't make sense. also the context can provide clues but maybe it just numerically fits that it fires without a strong semantic meaning.
make sure it is CONCISE AND CLEAR within 10 words or so.
Output only your label, nothing else. because i will directly display and read your output as the labels
"""


os.makedirs("./data/features", exist_ok=True)

all_batch_ids = []

for layer in config["layers"]:
    print(f"\nProcessing layer {layer}...")

    with open(pathconfig["feature_context"][layer], "r") as f:
        features = json.load(f)

    requests = []
    for fid, fdata in features.items():
        # hydrate contexts
        hydrated = copy.deepcopy(fdata)
        for ctx in hydrated["contexts"]:
            ex = examples.get(ctx["example_id"])
            if ex is None:
                continue
            ctx["input"] = ex["input"]
            ctx["output"] = ex["output"]
            del ctx["example_id"]

        user_msg = f"Feature {fid} | layer {layer}\n\n{json.dumps(hydrated, indent=2)}"

        requests.append({
            "custom_id": f"{layer}_{fid}",
            "params": {
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 50,
                "thinking": {"type": "disabled"},
                "system": ZERO_SHOT_PROMPT,
                "messages": [{"role": "user", "content": user_msg}]
            }
        })

    print(f"  Submitting {len(requests)} requests...")

    # split if too many requests (payload size limit)
    CHUNK_SIZE = 4000
    for i in range(0, len(requests), CHUNK_SIZE):
        chunk = requests[i:i+CHUNK_SIZE]
        print(f"    Chunk: {len(chunk)} requests...")
        batch = client.messages.batches.create(requests=chunk)
        all_batch_ids.append(batch.id)
        print(f"    Batch ID: {batch.id}")

with open(pathconfig["batch_ids"], "w") as f:
    f.write("\n".join(all_batch_ids))

print(f"\nAll batches submitted! {len(all_batch_ids)} total")
