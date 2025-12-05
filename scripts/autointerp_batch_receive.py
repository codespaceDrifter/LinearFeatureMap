# IMPORTANT: note that this could be EXPENSIVE!!! make sure to have enough credits.
import json
import anthropic
import time
from scripts.config import config, pathconfig

client = anthropic.Anthropic()

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

# collect all results: {layer: {fid: {interpretation}}}
results = {}

for batch_id in batch_ids:
    for result in client.messages.batches.results(batch_id):
        layer, fid = result.custom_id.split("_", 1)

        if layer not in results:
            results[layer] = {}

        if result.result.type == "succeeded":
            interp = ""
            for block in result.result.message.content:
                if block.type == "text":
                    interp = block.text.strip()
                    break

            results[layer][fid] = {
                "interpretation": interp,
            }
        else:
            print(f"Error: {result.custom_id} - {result.result}")

# write per-layer interpretation files
for layer in config["layers"]:
    layer_str = str(layer)
    if layer_str in results:
        with open(pathconfig["interpretations"][layer], "w") as f:
            json.dump(results[layer_str], f, indent=2)
        print(f"Layer {layer}: {len(results[layer_str])} interpretations saved")

total = sum(len(l) for l in results.values())
print(f"\nDone! {total} interpretations saved to ./data/features/")
