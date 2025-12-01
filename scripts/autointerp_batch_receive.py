import json
import anthropic
import time

client = anthropic.Anthropic()

# load batch id
with open("./data/batch_id.txt", "r") as f:
    batch_id = f.read().strip()

print(f"Checking batch: {batch_id}")

# poll until done
while True:
    batch = client.messages.batches.retrieve(batch_id)
    print(f"Status: {batch.processing_status} | Succeeded: {batch.request_counts.succeeded} | Processing: {batch.request_counts.processing} | Errored: {batch.request_counts.errored}")
    
    if batch.processing_status == "ended":
        break
    
    print("Waiting 60s...")
    time.sleep(60)

print("\nBatch complete! Retrieving results...")

# collect all results
results = {}

for result in client.messages.batches.results(batch_id):
    custom_id = result.custom_id
    layer, pos, fid = custom_id.split("_", 2)
    
    if result.result.type == "succeeded":
        msg = result.result.message
        interp = ""
        for block in msg.content:
            if block.type == "text":
                interp = block.text.strip()
                break
        
        is_interpretable = "uninterpretable" not in interp.lower()
        
        if layer not in results:
            results[layer] = {}
        if pos not in results[layer]:
            results[layer][pos] = {}
        
        results[layer][pos][fid] = {
            "interpretation": interp,
            "is_interpretable": is_interpretable
        }
    else:
        print(f"Error for {custom_id}: {result.result.type}")

with open("./data/batched_auto_interp.json", "w") as f:
    json.dump(results, f, indent=2)

total = sum(len(v) for v in results.values())
print(f"\nDone! {total} interpretations saved to ./data/batched_auto_interp.json")