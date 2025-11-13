"""
Extract activations from Gemma model for SAE training.

For each MLP layer, we extract PURE MLP activations using forward hooks:
1. mlp_input: Activation going INTO the MLP (after attention, before MLP)
2. mlp_output: Activation coming OUT of the MLP (after down_proj)
3. input_token_id: Token that produced this activation
4. predicted_token_id: Token predicted from this activation

This data is used to train:
- SAE: Sparse Autoencoder on all activations
- LFM: Linear Feature Map from input_features -> output_features

Implementation: Uses transformers library with forward hooks on model.model.layers[i].mlp
to capture pure MLP transformations (not full layer with attention).
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class ActivationSample:
    """Single sample of activations for one position in one sequence."""
    layer_idx: int              # Which MLP layer (0-25 for Gemma 3 1B)
    position: int               # Token position in sequence
    input_token_id: int         # Token ID that produced this activation
    predicted_token_id: int     # Token ID predicted from this activation
    mlp_input: np.ndarray       # Activation before MLP [hidden_dim]
    mlp_output: np.ndarray      # Activation after MLP [hidden_dim]


class ActivationExtractor:
    """Extract activations from Gemma for SAE training."""

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize extractor with model.

        Args:
            model_path: Path to model weights (e.g., "./weights")
            device: Device to run on ("cuda" or "cpu")
        """
        print(f"Loading model from {model_path}...")
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto",
            output_hidden_states=True  # CRITICAL: Get all layer outputs
        )
        self.model.eval()

        # Get model config - Gemma3 multimodal has nested text_config
        if hasattr(self.model.config, 'text_config'):
            text_config = self.model.config.text_config
            self.text_model = self.model.language_model  # Multimodal: text decoder is nested
            print("✓ Detected multimodal Gemma3 model")
        else:
            text_config = self.model.config
            self.text_model = self.model

        self.num_layers = text_config.num_hidden_layers
        self.hidden_dim = text_config.hidden_size

        print(f"✓ Model loaded: {self.num_layers} layers, hidden_dim={self.hidden_dim}")

    def extract_from_text(self, text: str) -> List[ActivationSample]:
        """
        Extract all activations from a single text.

        Args:
            text: Input text to process

        Returns:
            List of ActivationSample, one per (layer, position)
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]  # [seq_len]
        seq_len = input_ids.shape[0]

        samples = []

        # Storage for MLP activations captured by hooks
        mlp_captures = {}

        def make_hook(layer_idx):
            """Create hook function for specific layer."""
            def hook_fn(module, input, output):
                # input is tuple, output is tensor
                mlp_captures[layer_idx] = {
                    'input': input[0].detach().cpu().float(),  # [batch, seq_len, hidden_dim]
                    'output': output.detach().cpu().float()     # [batch, seq_len, hidden_dim]
                }
            return hook_fn

        # Register hooks on all MLP layers
        hooks = []
        for layer_idx in range(self.num_layers):
            hook = self.text_model.model.layers[layer_idx].mlp.register_forward_hook(
                make_hook(layer_idx)
            )
            hooks.append(hook)

        try:
            with torch.no_grad():
                # Forward pass - hooks will capture MLP activations
                outputs = self.model(
                    **inputs,
                    return_dict=True
                )

                logits = outputs.logits[0]  # [seq_len, vocab_size]

                # Process captured activations
                for layer_idx in range(self.num_layers):
                    mlp_data = mlp_captures[layer_idx]
                    mlp_in = mlp_data['input'][0]   # [seq_len, hidden_dim]
                    mlp_out = mlp_data['output'][0]  # [seq_len, hidden_dim]

                    # For each position in sequence
                    for pos in range(seq_len):
                        # Get input token at this position
                        input_token_id = input_ids[pos].item()

                        # Get predicted token at this position
                        predicted_token_id = torch.argmax(logits[pos]).item()

                        # Get MLP input/output for this position
                        sample = ActivationSample(
                            layer_idx=layer_idx,
                            position=pos,
                            input_token_id=input_token_id,
                            predicted_token_id=predicted_token_id,
                            mlp_input=mlp_in[pos].numpy(),   # [hidden_dim]
                            mlp_output=mlp_out[pos].numpy()  # [hidden_dim]
                        )
                        samples.append(sample)

        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()

        return samples

    def extract_from_dataset(
        self,
        texts: List[str],
        max_samples: int = None
    ) -> List[ActivationSample]:
        """
        Extract activations from multiple texts.

        Args:
            texts: List of text strings
            max_samples: Max number of samples to extract (None = all)

        Returns:
            List of all ActivationSamples
        """
        all_samples = []

        for i, text in enumerate(tqdm(texts, desc="Extracting activations")):
            samples = self.extract_from_text(text)
            all_samples.extend(samples)

            if max_samples and len(all_samples) >= max_samples:
                all_samples = all_samples[:max_samples]
                break

        print(f"✓ Extracted {len(all_samples)} activation samples")
        print(f"  - {self.num_layers} layers × {len(all_samples) // self.num_layers} positions")

        return all_samples

    def save_samples(self, samples: List[ActivationSample], output_path: str):
        """Save extracted samples to disk."""
        print(f"Saving {len(samples)} samples to {output_path}...")

        # Convert to numpy arrays for efficient storage
        data = {
            'layer_idx': np.array([s.layer_idx for s in samples], dtype=np.int32),
            'position': np.array([s.position for s in samples], dtype=np.int32),
            'input_token_id': np.array([s.input_token_id for s in samples], dtype=np.int32),
            'predicted_token_id': np.array([s.predicted_token_id for s in samples], dtype=np.int32),
            'mlp_input': np.stack([s.mlp_input for s in samples]),    # [N, hidden_dim]
            'mlp_output': np.stack([s.mlp_output for s in samples]),  # [N, hidden_dim]
        }

        np.savez_compressed(output_path, **data)
        print(f"✓ Saved to {output_path}")

    @staticmethod
    def load_samples(input_path: str) -> List[ActivationSample]:
        """Load samples from disk."""
        print(f"Loading samples from {input_path}...")
        data = np.load(input_path)

        samples = []
        for i in range(len(data['layer_idx'])):
            sample = ActivationSample(
                layer_idx=int(data['layer_idx'][i]),
                position=int(data['position'][i]),
                input_token_id=int(data['input_token_id'][i]),
                predicted_token_id=int(data['predicted_token_id'][i]),
                mlp_input=data['mlp_input'][i],
                mlp_output=data['mlp_output'][i]
            )
            samples.append(sample)

        print(f"✓ Loaded {len(samples)} samples")
        return samples


def demo():
    """Demo: Extract activations from sample text."""

    # Initialize extractor
    extractor = ActivationExtractor(
        model_path="../gemma3-1b/weights",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Sample texts
    texts = [
        "The capital of France is Paris.",
        "Machine learning is a subset of artificial intelligence.",
        "The quick brown fox jumps over the lazy dog."
    ]

    # Extract activations
    samples = extractor.extract_from_dataset(texts)

    # Show sample
    print("\nExample activation sample:")
    sample = samples[100]  # Random sample
    print(f"  Layer: {sample.layer_idx}")
    print(f"  Position: {sample.position}")
    print(f"  Input token: '{extractor.tokenizer.decode([sample.input_token_id])}'")
    print(f"  Predicted token: '{extractor.tokenizer.decode([sample.predicted_token_id])}'")
    print(f"  MLP input shape: {sample.mlp_input.shape}")
    print(f"  MLP output shape: {sample.mlp_output.shape}")
    print(f"  MLP input (first 10 dims): {sample.mlp_input[:10]}")
    print(f"  MLP output (first 10 dims): {sample.mlp_output[:10]}")

    # Save samples
    extractor.save_samples(samples, "activations.npz")

    # Load back
    loaded = extractor.load_samples("activations.npz")
    print(f"\n✓ Verified: Loaded {len(loaded)} samples")


if __name__ == "__main__":
    demo()
