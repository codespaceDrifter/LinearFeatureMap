# IMPORTANT: note that this could be EXPENSIVE!!! make sure to have enough credits.
import json
import anthropic
import time
from scripts.config import config, pathconfig

client = anthropic.Anthropic()

num_layers = config["num_layers"]

with open(pathconfig["batch_ids"], "r") as f:
    batch_ids = f.read().strip().split("\n")

print(f"Checking {len(batch_ids)} batches...")

# poll all until done
for batch_id in batch_ids:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(f"{batch_id[:20]}... | {batch.processing_status} | ok:{c.succeeded} err:{c.errored} exp:{c.expired}")

        if batch.processing_status == "ended":
            break

        time.sleep(60)

print("\nRetrieving results...")

# collect all results: {kind: {layer: {fid: {interpretation}}}}
# custom_id format: kind_layer_fid (e.g., "mlp_5_1234")
results_mlp = {}
results_att = {}

for batch_id in batch_ids:
    for result in client.messages.batches.results(batch_id):
        parts = result.custom_id.split("_", 2)
        if len(parts) != 3:
            print(f"Skipping malformed custom_id: {result.custom_id}")
            continue

        kind, layer, fid = parts

        if kind == "mlp":
            if layer not in results_mlp:
                results_mlp[layer] = {}
            target = results_mlp[layer]
        else:
            if layer not in results_att:
                results_att[layer] = {}
            target = results_att[layer]

        if result.result.type == "succeeded":
            interp = ""
            for block in result.result.message.content:
                if block.type == "text":
                    interp = block.text.strip()
                    break

            target[fid] = {
                "interpretation": interp,
            }
        else:
            print(f"Error: {result.custom_id} - {result.result}")

# write mlp interpretation files (layers 0 to 30)
for layer in range(num_layers - 1):
    layer_str = str(layer)
    if layer_str in results_mlp:
        with open(pathconfig["interpretations"]["mlp"][layer], "w") as f:
            json.dump(results_mlp[layer_str], f, indent=2)
        print(f"mlp[{layer}]: {len(results_mlp[layer_str])} interpretations saved")

# write att interpretation files (layers 1 to 31)
for layer in range(1, num_layers):
    layer_str = str(layer)
    if layer_str in results_att:
        with open(pathconfig["interpretations"]["att"][layer], "w") as f:
            json.dump(results_att[layer_str], f, indent=2)
        print(f"att[{layer}]: {len(results_att[layer_str])} interpretations saved")

total_mlp = sum(len(l) for l in results_mlp.values())
total_att = sum(len(l) for l in results_att.values())
print(f"\nDone! {total_mlp} mlp + {total_att} att = {total_mlp + total_att} interpretations saved")
