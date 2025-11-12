#!/usr/bin/env python3
"""
Simple chat interface for Gemma 3 1B using the official Google implementation.
"""

import sys
import os
import torch

# Add gemma_pytorch to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gemma_pytorch'))

from gemma import config as gemma_config
from gemma import model as gemma_model


def load_model(weights_dir="weights"):
    """Load the Gemma 3 1B model with weights."""
    print("Loading Gemma 3 1B model...")

    # Create config
    model_config = gemma_config.GemmaConfig(
        architecture=gemma_config.Architecture.GEMMA_3,
        vocab_size=256000,
        hidden_size=1152,
        intermediate_size=6912,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        quant=False,
        use_qk_norm=True,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
        attn_types=[
            gemma_config.AttentionType.LOCAL_SLIDING,
            gemma_config.AttentionType.LOCAL_SLIDING,
            gemma_config.AttentionType.LOCAL_SLIDING,
            gemma_config.AttentionType.LOCAL_SLIDING,
            gemma_config.AttentionType.LOCAL_SLIDING,
            gemma_config.AttentionType.GLOBAL,
        ],
        sliding_window_size=1024,
        rope_wave_length={
            gemma_config.AttentionType.LOCAL_SLIDING: 10000,
            gemma_config.AttentionType.GLOBAL: 10000,
        },
        rope_scaling_factor=1,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        tokenizer=os.path.join(weights_dir, "tokenizer.model"),
    )

    # Create embedding and model
    embedder = gemma_model.Embedding(model_config.vocab_size, model_config.hidden_size, model_config.quant)
    model = gemma_model.GemmaModel(model_config)
    sampler = gemma_model.Sampler(model_config.vocab_size, model_config)

    # Load weights
    print(f"Loading weights from {weights_dir}...")
    checkpoint_path = os.path.join(weights_dir, "model.safetensors")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(weights_dir, "pytorch_model.bin")

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        # Load into embedder and model
        embedder.load_state_dict({k.replace('text_token_embedder.', ''): v
                                  for k, v in state_dict.items()
                                  if k.startswith('text_token_embedder')}, strict=False)
        model.load_state_dict({k.replace('model.', ''): v
                              for k, v in state_dict.items()
                              if k.startswith('model')}, strict=False)
        print("✓ Weights loaded successfully!")
    else:
        print(f"Warning: Could not find weights at {checkpoint_path}")
        print("Model will run with random weights (for testing structure only)")

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedder = embedder.to(device)
    model = model.to(device)

    print(f"✓ Model loaded on {device}")
    print(f"✓ Ready to chat!")

    return embedder, model, sampler, model_config, device


def generate_response(prompt, embedder, model, sampler, config, device, max_new_tokens=100, temperature=0.7):
    """Generate a response to the prompt."""
    from gemma import tokenizer as gemma_tokenizer

    # Load tokenizer
    tok = gemma_tokenizer.Tokenizer(config.tokenizer)

    # Tokenize
    input_ids = tok.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    batch_size, seq_len = input_tensor.shape

    # Create KV caches
    kv_caches = []
    for _ in range(config.num_hidden_layers):
        k_cache = torch.zeros(
            (batch_size, seq_len + max_new_tokens, config.num_key_value_heads, config.head_dim),
            device=device,
            dtype=torch.float32
        )
        v_cache = torch.zeros(
            (batch_size, seq_len + max_new_tokens, config.num_key_value_heads, config.head_dim),
            device=device,
            dtype=torch.float32
        )
        kv_caches.append((k_cache, v_cache))

    # Precompute RoPE frequencies
    freqs_cis = {}
    freqs_cis[gemma_config.AttentionType.LOCAL_SLIDING] = gemma_model.precompute_freqs_cis(
        config.head_dim, (seq_len + max_new_tokens) * 2,
        theta=config.rope_wave_length[gemma_config.AttentionType.LOCAL_SLIDING]
    ).to(device)
    freqs_cis[gemma_config.AttentionType.GLOBAL] = gemma_model.precompute_freqs_cis(
        config.head_dim, (seq_len + max_new_tokens) * 2,
        theta=config.rope_wave_length[gemma_config.AttentionType.GLOBAL]
    ).to(device)

    # Generate tokens
    generated_ids = input_ids.copy()

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Get embeddings
            hidden_states = embedder(input_tensor)
            normalizer = torch.tensor(config.hidden_size**0.5, dtype=hidden_states.dtype, device=device)
            hidden_states = hidden_states * normalizer

            # Prepare masks and positions
            current_seq_len = input_tensor.shape[1]
            kv_write_indices = torch.arange(0, current_seq_len, dtype=torch.int64, device=device)

            # Create causal mask
            min_dtype = torch.finfo(torch.float32).min
            mask = torch.full((batch_size, 1, current_seq_len, seq_len + max_new_tokens), min_dtype, device=device)
            mask[:, :, :, :current_seq_len] = torch.triu(
                torch.full((current_seq_len, current_seq_len), min_dtype, device=device),
                diagonal=1
            )
            local_mask = mask

            # Forward through model
            hidden_states = model(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                kv_write_indices=kv_write_indices,
                kv_caches=kv_caches,
                mask=mask,
                local_mask=local_mask,
            )

            # Sample next token
            embedder_weight = embedder.weight
            output_positions = torch.tensor([current_seq_len - 1], dtype=torch.int64, device=device)
            temperatures = torch.tensor([temperature], dtype=torch.float32, device=device)
            top_ps = torch.tensor([0.95], dtype=torch.float32, device=device)
            top_ks = torch.tensor([64], dtype=torch.int64, device=device)

            next_token, _ = sampler(
                embedding=embedder_weight,
                hidden_states=hidden_states,
                output_positions=output_positions,
                temperatures=temperatures,
                top_ps=top_ps,
                top_ks=top_ks,
            )

            # Check for EOS
            if next_token.item() == tok.eos_id:
                break

            # Append to generated
            generated_ids.append(next_token.item())
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

    # Decode
    response = tok.decode(generated_ids)
    return response


def main():
    print("="*60)
    print("Gemma 3 1B Chat - Official Google Implementation")
    print("="*60)
    print()

    # Load model
    embedder, model, sampler, config, device = load_model()

    print()
    print("="*60)
    print("Chat started! Type 'quit' or 'exit' to stop.")
    print("="*60)
    print()

    while True:
        # Get user input
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        # Generate response
        print("Gemma: ", end="", flush=True)
        try:
            response = generate_response(
                user_input,
                embedder,
                model,
                sampler,
                config,
                device,
                max_new_tokens=100,
                temperature=0.7
            )
            # Remove the prompt from response
            if response.startswith(user_input):
                response = response[len(user_input):].strip()
            print(response)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print()


if __name__ == "__main__":
    main()
