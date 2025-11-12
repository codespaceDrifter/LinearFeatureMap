"""
Gemma 3 1B Model - Full PyTorch Implementation from Scratch
This is the actual model architecture, not a wrapper around transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class Gemma3RMSNorm(nn.Module):
    """RMSNorm normalization layer used throughout Gemma 3."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """Root Mean Square normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for positional encoding.
    Used in both queries and keys of attention.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        """
        Args:
            x: input tensor
            seq_len: sequence length
        Returns:
            cos, sin: cosine and sine of position encodings
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        # Create position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)

        # Compute frequencies
        freqs = torch.outer(t, self.inv_freq)
        # Concatenate to match head_dim
        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos(), emb.sin()


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Apply rotary position embeddings to query and key tensors.
    """
    # Expand cos and sin to match q, k dimensions
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class Gemma3Attention(nn.Module):
    """
    Grouped-Query Attention (GQA) with QK normalization.
    Gemma 3 uses local and global attention patterns.
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        is_local: bool = False,
        sliding_window: int = 1024,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_local = is_local
        self.sliding_window = sliding_window

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # QK normalization (replaces soft-capping from Gemma 2)
        self.q_norm = Gemma3RMSNorm(head_dim, eps=1e-6)
        self.k_norm = Gemma3RMSNorm(head_dim, eps=1e-6)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)

        # Apply QK normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose for attention computation
        query_states = query_states.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=seq_length)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat k/v heads for grouped-query attention
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply local attention mask if this is a local attention layer
        if self.is_local and seq_length > 1:
            # Create sliding window mask
            local_mask = torch.ones(seq_length, seq_length, device=hidden_states.device)
            local_mask = torch.triu(local_mask, diagonal=-self.sliding_window)
            local_mask = torch.tril(local_mask, diagonal=0)
            local_mask = local_mask.bool()
            attn_weights = attn_weights.masked_fill(~local_mask, float('-inf'))

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Gemma3MLP(nn.Module):
    """
    MLP layer with gated projection.
    Uses PytorchGELUTanh activation.
    """

    def __init__(self, hidden_size: int = 1152, intermediate_size: int = 6912):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # Gated activation: gate_proj(x) * GELU(up_proj(x))
        gate = F.gelu(self.gate_proj(x), approximate='tanh')  # PytorchGELUTanh
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Gemma3DecoderLayer(nn.Module):
    """
    Single Gemma 3 decoder layer with pre-norm and post-norm.
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 6912,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        is_local: bool = False,
    ):
        super().__init__()

        # Attention
        self.self_attn = Gemma3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            is_local=is_local,
        )

        # MLP
        self.mlp = Gemma3MLP(hidden_size, intermediate_size)

        # Normalization layers (Gemma 3 uses 4 RMSNorm per layer)
        self.input_layernorm = Gemma3RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = Gemma3RMSNorm(hidden_size, eps=1e-6)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(hidden_size, eps=1e-6)
        self.post_feedforward_layernorm = Gemma3RMSNorm(hidden_size, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.

        Returns:
            hidden_states: output of this layer
        """
        # Self-attention with pre-norm and post-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with pre-norm and post-norm
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3Model(nn.Module):
    """
    Complete Gemma 3 1B model implementation.

    Architecture:
    - 26 decoder layers
    - Hidden size: 1152
    - Intermediate size: 6912
    - 8 attention heads, 2 key-value heads (GQA)
    - Alternating local (5) and global (1) attention pattern
    - Vocab size: 256,000
    """

    def __init__(
        self,
        vocab_size: int = 256000,
        hidden_size: int = 1152,
        intermediate_size: int = 6912,
        num_hidden_layers: int = 26,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        max_position_embeddings: int = 32768,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Decoder layers with alternating local/global attention
        # Pattern: 5 local layers, then 1 global layer, repeating
        self.layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            # Every 6th layer (0-indexed: 5, 11, 17, 23) is global
            is_local = (i % 6) != 5
            layer = Gemma3DecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                is_local=is_local,
            )
            self.layers.append(layer)

        # Final layer norm
        self.norm = Gemma3RMSNorm(hidden_size, eps=1e-6)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask
            position_ids: Position IDs

        Returns:
            hidden_states: Final hidden states
            all_hidden_states: List of hidden states from each layer
        """
        batch_size, seq_length = input_ids.shape

        # Get token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Store all layer outputs for analysis
        all_hidden_states = [hidden_states]

        # Process through each decoder layer
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            all_hidden_states.append(hidden_states)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states, all_hidden_states


class Gemma3ForCausalLM(nn.Module):
    """
    Gemma 3 1B for causal language modeling.
    Adds language modeling head to the base model.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = Gemma3Model(**kwargs)
        self.vocab_size = self.model.vocab_size

        # Language modeling head (output projection)
        self.lm_head = nn.Linear(self.model.hidden_size, self.vocab_size, bias=False)

        # Tie weights with input embeddings (common practice)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for language modeling.

        Returns:
            logits: Output logits [batch_size, seq_length, vocab_size]
            all_hidden_states: Hidden states from all layers
            loss: If labels provided, returns cross-entropy loss
        """
        # Get hidden states from base model
        hidden_states, all_hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )

        return {
            'logits': logits,
            'all_hidden_states': all_hidden_states,
            'loss': loss,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        """
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(input_ids)
            logits = outputs['logits']

            # Get last token logits
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids


def create_gemma3_1b():
    """Create a Gemma 3 1B model with default configuration."""
    return Gemma3ForCausalLM(
        vocab_size=256000,
        hidden_size=1152,
        intermediate_size=6912,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=32768,
    )
