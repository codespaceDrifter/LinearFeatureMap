"""
Gemma 3 1B Layer-by-Layer Inference Script
This script allows you to run the Gemma 3 1B model layer by layer
and access activations at each stage for analysis and manipulation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import numpy as np


class LayerByLayerGemma:
    def __init__(self, model_id: str = "google/gemma-3-1b-pt", device: str = "auto"):
        """
        Initialize the Gemma model for layer-by-layer inference.

        Args:
            model_id: HuggingFace model ID (default: gemma-3-1b-pt for pretrained)
            device: Device to load model on ('auto', 'cuda', 'cpu')
        """
        print(f"Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            output_hidden_states=True,  # Enable hidden state outputs
        )
        self.model.eval()  # Set to evaluation mode

        # Get model architecture info
        self.config = self.model.config
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size

        print(f"Model loaded successfully!")
        print(f"Number of layers: {self.num_layers}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Intermediate size: {self.config.intermediate_size}")
        print(f"Number of attention heads: {self.config.num_attention_heads}")

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize input text."""
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get the initial token embeddings."""
        return self.model.model.embed_tokens(input_ids)

    def run_layer_by_layer(self, text: str, hook_fn=None) -> Dict:
        """
        Run inference layer by layer, collecting activations.

        Args:
            text: Input text to process
            hook_fn: Optional function to modify activations at each layer
                     Signature: hook_fn(layer_idx, hidden_states) -> modified_hidden_states

        Returns:
            Dict containing:
                - input_ids: tokenized input
                - embeddings: initial embeddings
                - layer_outputs: list of activations from each layer
                - final_logits: output logits
        """
        print(f"\nProcessing: '{text}'")

        # Tokenize
        input_ids = self.tokenize(text)
        print(f"Tokens: {input_ids.shape}")

        # Get initial embeddings
        hidden_states = self.get_embeddings(input_ids)
        print(f"Initial embeddings shape: {hidden_states.shape}")

        # Store activations
        activations = {
            'input_ids': input_ids,
            'embeddings': hidden_states.clone(),
            'layer_outputs': [],
            'attention_outputs': []
        }

        # Process through each layer
        with torch.no_grad():
            # Get position embeddings and other inputs needed
            attention_mask = torch.ones_like(input_ids)
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=self.model.device)
            position_ids = position_ids.unsqueeze(0)

            # Process through each transformer layer
            for layer_idx, layer in enumerate(self.model.model.layers):
                print(f"\nProcessing layer {layer_idx + 1}/{self.num_layers}")

                # Apply hook function if provided
                if hook_fn is not None:
                    hidden_states = hook_fn(layer_idx, hidden_states)

                # Run through the layer
                layer_output = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

                # layer_output is a tuple (hidden_states, ...)
                hidden_states = layer_output[0]

                # Store layer output
                activations['layer_outputs'].append({
                    'layer_idx': layer_idx,
                    'hidden_states': hidden_states.clone(),
                    'shape': hidden_states.shape,
                    'mean': hidden_states.mean().item(),
                    'std': hidden_states.std().item(),
                })

                print(f"  Output shape: {hidden_states.shape}")
                print(f"  Mean: {hidden_states.mean().item():.4f}, Std: {hidden_states.std().item():.4f}")

            # Apply final layer norm
            hidden_states = self.model.model.norm(hidden_states)
            print(f"\nAfter final layer norm:")
            print(f"  Shape: {hidden_states.shape}")

            # Get logits
            logits = self.model.lm_head(hidden_states)
            activations['final_logits'] = logits

            print(f"\nFinal logits shape: {logits.shape}")

        return activations

    def run_with_intermediate_modifications(self, text: str, modify_fn):
        """
        Run inference with custom modifications to activations.

        Example modify_fn that scales activations at layer 10:
            def modify_fn(layer_idx, hidden_states):
                if layer_idx == 10:
                    return hidden_states * 1.5
                return hidden_states
        """
        return self.run_layer_by_layer(text, hook_fn=modify_fn)

    def decode_logits(self, logits: torch.Tensor, top_k: int = 5) -> List[str]:
        """Decode logits to tokens."""
        # Get the last token's logits
        last_token_logits = logits[0, -1, :]
        top_tokens = torch.topk(last_token_logits, top_k)

        results = []
        for score, token_id in zip(top_tokens.values, top_tokens.indices):
            token = self.tokenizer.decode([token_id])
            results.append((token, score.item()))

        return results

    def generate_token_by_token(self, text: str, max_new_tokens: int = 10, temperature: float = 1.0):
        """
        Generate text token by token with access to layer outputs.
        """
        print(f"\n{'='*60}")
        print(f"Generating with layer-by-layer access")
        print(f"{'='*60}")

        input_ids = self.tokenize(text)
        generated_text = text

        for step in range(max_new_tokens):
            print(f"\n--- Generation step {step + 1}/{max_new_tokens} ---")

            # Run through all layers
            result = self.run_layer_by_layer(generated_text)

            # Get next token
            logits = result['final_logits'][0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Decode and append
            next_token = self.tokenizer.decode(next_token_id)
            generated_text += next_token

            print(f"Generated token: '{next_token}'")
            print(f"Current text: {generated_text}")

        return generated_text


def main():
    """Example usage of the LayerByLayerGemma class."""

    print("="*60)
    print("Gemma 3 1B Layer-by-Layer Inference")
    print("="*60)

    # Initialize model
    model = LayerByLayerGemma(
        model_id="google/gemma-3-1b-pt",  # Use -pt for pretrained, -it for instruction-tuned
        device="auto"
    )

    # Example 1: Basic layer-by-layer inference
    print("\n" + "="*60)
    print("Example 1: Basic Layer-by-Layer Inference")
    print("="*60)

    text = "The capital of France is"
    result = model.run_layer_by_layer(text)

    print(f"\n\nActivation statistics per layer:")
    for layer_output in result['layer_outputs']:
        print(f"Layer {layer_output['layer_idx']:2d}: "
              f"mean={layer_output['mean']:7.4f}, "
              f"std={layer_output['std']:6.4f}")

    # Get top predictions
    top_predictions = model.decode_logits(result['final_logits'], top_k=5)
    print(f"\nTop 5 predictions for next token:")
    for token, score in top_predictions:
        print(f"  '{token}': {score:.4f}")

    # Example 2: Modify activations at a specific layer
    print("\n" + "="*60)
    print("Example 2: Modifying Activations at Layer 10")
    print("="*60)

    def amplify_layer_10(layer_idx, hidden_states):
        """Amplify activations at layer 10 by 1.5x."""
        if layer_idx == 10:
            print(f"  >> Amplifying activations by 1.5x at layer {layer_idx}")
            return hidden_states * 1.5
        return hidden_states

    modified_result = model.run_with_intermediate_modifications(text, amplify_layer_10)
    modified_predictions = model.decode_logits(modified_result['final_logits'], top_k=5)

    print(f"\nTop 5 predictions after modification:")
    for token, score in modified_predictions:
        print(f"  '{token}': {score:.4f}")

    # Example 3: Generate text token by token
    print("\n" + "="*60)
    print("Example 3: Token-by-Token Generation")
    print("="*60)

    generated = model.generate_token_by_token(
        "Once upon a time",
        max_new_tokens=5,
        temperature=0.7
    )

    print(f"\n\nFinal generated text: {generated}")


if __name__ == "__main__":
    main()
