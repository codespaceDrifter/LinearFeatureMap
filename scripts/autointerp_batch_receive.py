import json
import anthropic
import time

client = anthropic.Anthropic()

with open("./data/labels/batch_ids.txt", "r") as f:
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

# collect all results: {layer: {pos: {fid: {interpretation, is_interpretable}}}}
results = {}

for batch_id in batch_ids:
    for result in client.messages.batches.results(batch_id):
        layer, pos, fid = result.custom_id.split("_", 2)
        
        if layer not in results:
            results[layer] = {}
        if pos not in results[layer]:
            results[layer][pos] = {}
        
        if result.result.type == "succeeded":
            interp = ""
            for block in result.result.message.content:
                if block.type == "text":
                    interp = block.text.strip()
                    break
            
            results[layer][pos][fid] = {
                "interpretation": interp,
                "is_interpretable": "uninterpretable" not in interp.lower()
            }
        else:
            print(f"Error: {result.custom_id} - {result.result.type}")

with open("./data/labels/interpretations.json", "w") as f:
    json.dump(results, f, indent=2)

total = sum(len(p) for l in results.values() for p in l.values())
print(f"\nDone! {total} interpretations saved to ./data/labels/interpretations.json")