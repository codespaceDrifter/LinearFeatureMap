import torch.nn as nn

class LFM(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim, bias=False)
    
    def forward(self, f_in):
        return self.linear(f_in)