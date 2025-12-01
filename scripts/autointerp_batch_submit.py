import json
import copy
import anthropic
import os

LAYERS = [8, 16, 24]
client = anthropic.Anthropic()

# load examples
examples = {}
with open("./data/examples.jsonl", "r") as f:
    for line in f:
        ex = json.loads(line)
        examples[ex["id"]] = {"input": ex["input"], "output": ex["output"]}

SYSTEM_PROMPT = """
Hi claudy! You will help me label features for mechanistic interpretability of Sparse Autoencoder.
You'll see:
1. Contexts where this feature activated (we trained on model outputs in the alpaca dataset) Context includes:
Activations. which are the tokens in the answer that led this feature to fire. formated as [index] , 'token' , activation value
the input question in the dataset
the output answer in the model wrote

2. Top decoded tokens: top 10 dot product of decoder weights in SAE for that feature with all token embeddings (both normalized)
formatted as first the list of top decoded tokens and then their corresponding scores

Your job:
first, critically think a bit with your thinking tokens and determine what this feature means. consider all provided information do not be lazy and just look at one and pick that. 
make sure the feature interpretation you draft actually explains many or most of the contexts pretty well. top decoded could provide clues but the middle layers might not be actually 
embedding space so maybe they won't make sense. also the context can provide clues but maybe it just numerically fits that it fires without a strong semantic meaning.
You're smart. make your best decision.  
now you are done thinking you can write your label. make sure it is CONCISE AND CLEAR within 10 words or so. Optionally after your label if this feature requires more explanation, you can
put a colon : and then provide more explanation. 
If no clear pattern exists, which many features might have, do not force one, just say:
UNINTERPRETABLE somewhere in your output answer(just the word uninterpetable case doesn't matter). and i will programmatically ignore them for later use.
Output only your label, nothing else. no courtesy or 'ok let me start now' or reasoning steps keep them in thinking tokens. 
because i will directly display and read your output as the labels"""

os.makedirs("./data/labels", exist_ok=True)

all_batch_ids = []

for layer in LAYERS:
    for pos in ["pre", "post"]:
        print(f"\nProcessing layer_{layer}_{pos}...")
        
        with open(f"./data/features/layer_{layer}_{pos}.json", "r") as f:
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
            
            user_msg = f"Feature {fid} | layer {layer} {pos}-MLP\n\n{json.dumps(hydrated, indent=2)}"
            
            requests.append({
                "custom_id": f"{layer}_{pos}_{fid}",
                "params": {
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 2200,
                    "thinking": {"type": "enabled", "budget_tokens": 2000},
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_msg}]
                }
            })
        
        print(f"  Submitting {len(requests)} requests...")
        
        # split if too many requests (payload size limit)
        CHUNK_SIZE = 5000
        for i in range(0, len(requests), CHUNK_SIZE):
            chunk = requests[i:i+CHUNK_SIZE]
            print(f"    Chunk: {len(chunk)} requests...")
            batch = client.messages.batches.create(requests=chunk)
            all_batch_ids.append(batch.id)
            print(f"    Batch ID: {batch.id}")

with open("./data/labels/batch_ids.txt", "w") as f:
    f.write("\n".join(all_batch_ids))

print(f"\nAll batches submitted! {len(all_batch_ids)} total")