"""
Simple test to verify the model loads and runs.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Testing Gemma 3 1B model loading...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Try to load the model
model_id = "google/gemma-3-1b-pt"
print(f"\nLoading model: {model_id}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("✓ Tokenizer loaded successfully")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("✓ Model loaded successfully")

    # Print model info
    print(f"\nModel config:")
    print(f"  - Layers: {model.config.num_hidden_layers}")
    print(f"  - Hidden size: {model.config.hidden_size}")
    print(f"  - Attention heads: {model.config.num_attention_heads}")
    print(f"  - Intermediate size: {model.config.intermediate_size}")

    # Test tokenization
    text = "Hello, world!"
    inputs = tokenizer(text, return_tensors="pt")
    print(f"\nTest tokenization: '{text}'")
    print(f"  Token IDs: {inputs['input_ids']}")
    print(f"  Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")

    print("\n✓ All tests passed!")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
