from datasets import load_dataset
import os

os.makedirs("./data", exist_ok=True)

# download alpaca - it only has train split actually
dataset = load_dataset("tatsu-lab/alpaca")

# save locally
dataset.save_to_disk("./data/alpaca")

print(f"Splits: {list(dataset.keys())}")
print(f"Train size: {len(dataset['train'])}")
print(f"Columns: {dataset['train'].column_names}")
print(f"\nExample:")
print(dataset['train'][0])