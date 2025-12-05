import torch
import torch.nn as nn

# meant to be created from other trained SAEs and a creation script, not meant to be trained since activations would be too large
class MergedSAE(nn.Module):
    """
    Merged SAE that processes multiple layers.

    Input:  (batch, layers, embed_dim)
    Output: (batch, layers, hidden_dim) for encode
            (batch, layers, embed_dim) for decode
    """

    def __init__(self, num_layers, embed_dim, hidden_dim):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # encoder: (layers, embed_dim, hidden_dim)
        self.encoder_weight = nn.Parameter(torch.zeros(num_layers, embed_dim, hidden_dim))
        self.encoder_bias = nn.Parameter(torch.zeros(num_layers, hidden_dim))

        # decoder: (layers, hidden_dim, embed_dim)
        self.decoder_weight = nn.Parameter(torch.zeros(num_layers, hidden_dim, embed_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(num_layers, embed_dim))

    def encode(self, x):
        # x: (batch, layers, embed_dim)
        x = x.unsqueeze(-2)  # (batch, layers, 1, embed_dim)
        z = torch.matmul(x, self.encoder_weight)  # (batch, layers, 1, hidden_dim)
        z = z.squeeze(-2)  # (batch, layers, hidden_dim)
        z = z + self.encoder_bias  # broadcasts (layers, hidden_dim)
        z = torch.relu(z)
        return z

    def decode(self, z):
        # z: (batch, layers, hidden_dim)
        z = z.unsqueeze(-2)  # (batch, layers, 1, hidden_dim)
        x = torch.matmul(z, self.decoder_weight)  # (batch, layers, 1, embed_dim)
        x = x.squeeze(-2)  # (batch, layers, embed_dim)
        x = x + self.decoder_bias  # broadcasts (layers, embed_dim)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
