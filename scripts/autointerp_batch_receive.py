import json
import anthropic
import time
import os

LAYERS = [8, 16, 24]
client = anthropic.Anthropic()

for layer in LAYERS:
    for pos in ["pre", "post"]:
        key_file = f"./data/labels/batch_request_key/layer_{layer}_{pos}.txt"
        out_file = f"./data/labels/layer_{layer}_{pos}.json"
        
        if not os.path.exists(key_file):
            print(f"Skipping layer_{layer}_{pos} - no batch key")
            continue
        
        with open(key_file, "r") as f:
            batch_id = f.read().strip()
        
        print(f"\nlayer_{layer}_{pos} (batch: {batch_id})")
        
        # poll until done
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            print(f"  Status: {batch.processing_status} | Succeeded: {batch.request_counts.succeeded} | Processing: {batch.request_counts.processing}")
            
            if batch.processing_status == "ended":
                break
            
            print("  Waiting 60s...")
            time.sleep(60)
        
        # collect results
        results = {}
        for result in client.messages.batches.results(batch_id):
            fid = result.custom_id
            
            if result.result.type == "succeeded":
                msg = result.result.message
                interp = ""
                for block in msg.content:
                    if block.type == "text":
                        interp = block.text.strip()
                        break
                
                is_interpretable = "uninterpretable" not in interp.lower()
                
                results[fid] = {
                    "interpretation": interp,
                    "is_interpretable": is_interpretable
                }
            else:
                print(f"  Error for {fid}: {result.result.type}")
        
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"  Saved {len(results)} interpretations to {out_file}")

print("\nDone!")