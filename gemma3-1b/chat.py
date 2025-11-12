#!/usr/bin/env python3
"""
Simple chat interface for Gemma 3 1B.
Just run: python chat.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def load_model(weights_path="./weights"):
    """Load Gemma 3 1B model and tokenizer."""
    print("Loading Gemma 3 1B...")
    print(f"  Weights: {weights_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(weights_path)
    model = AutoModelForCausalLM.from_pretrained(
        weights_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    print("✓ Model loaded!\n")
    return model, tokenizer, device

def chat(model, tokenizer, device):
    """Interactive chat loop."""
    print("="*60)
    print("Gemma 3 1B Chat")
    print("="*60)
    print("Type your message and press Enter. Type 'quit' to exit.\n")

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Generate response
        print("Gemma: ", end="", flush=True)

        inputs = tokenizer(user_input, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and print (skip the input prompt)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(response)
        print()

def main():
    # Check if weights exist
    if not os.path.exists("./weights"):
        print("ERROR: weights/ folder not found!")
        print("\nPlease download the model weights first:")
        print("  huggingface-cli download google/gemma-3-1b-pt --local-dir weights/")
        return

    # Load model
    model, tokenizer, device = load_model()

    # Start chatting
    chat(model, tokenizer, device)

if __name__ == "__main__":
    main()
