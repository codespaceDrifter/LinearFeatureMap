import anthropic
client = anthropic.Anthropic()

# cancel all batches
with open("./data/labels/batch_ids.txt", "r") as f:
    batch_ids = f.read().strip().split("\n")

for batch_id in batch_ids:
    try:
        client.messages.batches.cancel(batch_id)
        print(f"Cancelled: {batch_id}")
    except Exception as e:
        print(f"Failed {batch_id}: {e}")