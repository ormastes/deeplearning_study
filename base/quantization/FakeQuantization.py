import torch
import torch.nn as nn
import math

class FakeQuantization(nn.Module):
    def __init__(self, scale=1.0, zero_point=0.0):
        super(FakeQuantization, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.zero_point = nn.Parameter(torch.tensor(zero_point), requires_grad=True)

    def forward(self, x):
        q_x = (x / self.scale + self.zero_point).round().to(torch.int32)
        return q_x, self.scale

    def dequantize(self, q_x):
        return (q_x - self.zero_point) * self.scale