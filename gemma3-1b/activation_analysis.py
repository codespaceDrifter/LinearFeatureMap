"""
Example script for analyzing and manipulating activations in Gemma 3 1B.
This demonstrates common use cases for layer-by-layer access.
"""

import torch
import numpy as np
from layer_by_layer_inference import LayerByLayerGemma


class ActivationAnalyzer:
    """Tools for analyzing model activations."""

    def __init__(self, model: LayerByLayerGemma):
        self.model = model

    def extract_all_activations(self, text: str) -> dict:
        """
        Extract activations from all layers for a given input.
        Returns a structured dict with all activation tensors.
        """
        result = self.model.run_layer_by_layer(text)

        activations = {
            'text': text,
            'embeddings': result['embeddings'].cpu().numpy(),
            'layers': []
        }

        for layer_out in result['layer_outputs']:
            activations['layers'].append({
                'layer_idx': layer_out['layer_idx'],
                'activations': layer_out['hidden_states'].cpu().numpy(),
                'mean': layer_out['mean'],
                'std': layer_out['std']
            })

        activations['logits'] = result['final_logits'].cpu().numpy()

        return activations

    def compare_activations(self, text1: str, text2: str):
        """
        Compare activations between two different inputs.
        Useful for understanding how different inputs activate the model.
        """
        print(f"Comparing activations:")
        print(f"  Text 1: '{text1}'")
        print(f"  Text 2: '{text2}'")
        print()

        result1 = self.model.run_layer_by_layer(text1)
        result2 = self.model.run_layer_by_layer(text2)

        print(f"\n{'Layer':<8} {'Text1 Mean':<12} {'Text2 Mean':<12} {'Difference':<12} {'Cosine Sim':<12}")
        print("-" * 60)

        for i, (layer1, layer2) in enumerate(zip(result1['layer_outputs'], result2['layer_outputs'])):
            h1 = layer1['hidden_states'].flatten()
            h2 = layer2['hidden_states'].flatten()

            mean_diff = abs(layer1['mean'] - layer2['mean'])

            # Compute cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(h1, h2, dim=0).item()

            print(f"{i:<8} {layer1['mean']:<12.4f} {layer2['mean']:<12.4f} "
                  f"{mean_diff:<12.4f} {cos_sim:<12.4f}")

    def find_most_active_neurons(self, text: str, layer_idx: int, top_k: int = 10):
        """
        Find the most active neurons in a specific layer.
        """
        result = self.model.run_layer_by_layer(text)
        layer_output = result['layer_outputs'][layer_idx]
        activations = layer_output['hidden_states'][0, -1, :]  # Last token

        # Get top-k most active neurons
        top_values, top_indices = torch.topk(torch.abs(activations), top_k)

        print(f"\nTop {top_k} most active neurons in layer {layer_idx}:")
        print(f"Input: '{text}'")
        print()
        for i, (idx, val) in enumerate(zip(top_indices, top_values)):
            print(f"{i+1}. Neuron {idx.item():<6} | Activation: {val.item():.4f}")

        return top_indices, top_values

    def track_token_representations(self, text: str, token_position: int = -1):
        """
        Track how a specific token's representation changes through the layers.
        """
        result = self.model.run_layer_by_layer(text)

        print(f"\nTracking token at position {token_position} through layers:")
        print(f"Text: '{text}'")
        print()

        print(f"{'Layer':<8} {'Mean':<12} {'Std':<12} {'Max':<12} {'Min':<12}")
        print("-" * 60)

        for layer_out in result['layer_outputs']:
            token_repr = layer_out['hidden_states'][0, token_position, :]
            mean = token_repr.mean().item()
            std = token_repr.std().item()
            max_val = token_repr.max().item()
            min_val = token_repr.min().item()

            print(f"{layer_out['layer_idx']:<8} {mean:<12.4f} {std:<12.4f} "
                  f"{max_val:<12.4f} {min_val:<12.4f}")


class ActivationInterventions:
    """Examples of intervening on activations during inference."""

    def __init__(self, model: LayerByLayerGemma):
        self.model = model

    def amplify_layer(self, text: str, layer_idx: int, scale: float = 1.5):
        """Amplify all activations at a specific layer."""
        print(f"\nAmplifying layer {layer_idx} by {scale}x")

        def hook(idx, hidden_states):
            if idx == layer_idx:
                return hidden_states * scale
            return hidden_states

        result = self.model.run_with_intermediate_modifications(text, hook)
        predictions = self.model.decode_logits(result['final_logits'], top_k=5)

        print("\nTop predictions:")
        for token, score in predictions:
            print(f"  '{token}': {score:.4f}")

        return result

    def suppress_neurons(self, text: str, layer_idx: int, neuron_indices: list):
        """Zero out specific neurons in a layer."""
        print(f"\nSuppressing neurons {neuron_indices} in layer {layer_idx}")

        def hook(idx, hidden_states):
            if idx == layer_idx:
                modified = hidden_states.clone()
                for neuron_idx in neuron_indices:
                    modified[:, :, neuron_idx] = 0
                return modified
            return hidden_states

        result = self.model.run_with_intermediate_modifications(text, hook)
        predictions = self.model.decode_logits(result['final_logits'], top_k=5)

        print("\nTop predictions after suppression:")
        for token, score in predictions:
            print(f"  '{token}': {score:.4f}")

        return result

    def add_noise(self, text: str, layer_idx: int, noise_scale: float = 0.1):
        """Add Gaussian noise to activations at a specific layer."""
        print(f"\nAdding noise (scale={noise_scale}) to layer {layer_idx}")

        def hook(idx, hidden_states):
            if idx == layer_idx:
                noise = torch.randn_like(hidden_states) * noise_scale
                return hidden_states + noise
            return hidden_states

        result = self.model.run_with_intermediate_modifications(text, hook)
        predictions = self.model.decode_logits(result['final_logits'], top_k=5)

        print("\nTop predictions with noise:")
        for token, score in predictions:
            print(f"  '{token}': {score:.4f}")

        return result

    def activation_steering(self, text: str, layer_idx: int, steering_vector: torch.Tensor):
        """
        Add a steering vector to activations at a specific layer.
        This can be used to guide the model toward certain behaviors.
        """
        print(f"\nSteering activations at layer {layer_idx}")

        def hook(idx, hidden_states):
            if idx == layer_idx:
                # Add steering vector to all tokens
                return hidden_states + steering_vector
            return hidden_states

        result = self.model.run_with_intermediate_modifications(text, hook)
        predictions = self.model.decode_logits(result['final_logits'], top_k=5)

        print("\nTop predictions after steering:")
        for token, score in predictions:
            print(f"  '{token}': {score:.4f}")

        return result


def main():
    """Demo of activation analysis and interventions."""

    print("="*60)
    print("Gemma 3 1B - Activation Analysis Examples")
    print("="*60)

    # Initialize model
    model = LayerByLayerGemma(model_id="google/gemma-3-1b-pt")
    analyzer = ActivationAnalyzer(model)
    interventions = ActivationInterventions(model)

    # Example 1: Extract all activations
    print("\n" + "="*60)
    print("Example 1: Extract All Activations")
    print("="*60)

    text = "The quick brown fox"
    activations = analyzer.extract_all_activations(text)
    print(f"\nExtracted activations for: '{text}'")
    print(f"Number of layers: {len(activations['layers'])}")
    print(f"Embedding shape: {activations['embeddings'].shape}")

    # Example 2: Compare activations
    print("\n" + "="*60)
    print("Example 2: Compare Two Inputs")
    print("="*60)

    analyzer.compare_activations(
        "The capital of France is",
        "The capital of Germany is"
    )

    # Example 3: Find most active neurons
    print("\n" + "="*60)
    print("Example 3: Most Active Neurons")
    print("="*60)

    analyzer.find_most_active_neurons(
        "Machine learning is",
        layer_idx=10,
        top_k=5
    )

    # Example 4: Track token through layers
    print("\n" + "="*60)
    print("Example 4: Track Token Representations")
    print("="*60)

    analyzer.track_token_representations("Hello world", token_position=-1)

    # Example 5: Amplify a layer
    print("\n" + "="*60)
    print("Example 5: Amplify Layer Activations")
    print("="*60)

    interventions.amplify_layer("The meaning of life is", layer_idx=12, scale=2.0)

    # Example 6: Add noise
    print("\n" + "="*60)
    print("Example 6: Add Noise to Activations")
    print("="*60)

    interventions.add_noise("Once upon a time", layer_idx=5, noise_scale=0.2)

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
