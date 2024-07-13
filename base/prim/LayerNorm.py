import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        # weight has both scale and shift
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        size = x.shape[-1]
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False to use the same normalization as GPT-2
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / (var + self.eps).sqrt()
        return norm_x * self.scale[:size] + self.shift[:size]