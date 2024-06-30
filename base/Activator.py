import torch
from torch import nn


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((x * (1 + 0.044715 * x * x))))