import torch
from torch import nn
import math
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.rand(self.out_features, self.in_features))
        self.bias_enabled = bias
        if self.bias_enabled:
            self.bias = torch.nn.Parameter(torch.rand(self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # original linear use kaiming_uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_enabled:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def dim_match_forward(self, inputs):
        # `y = xA^T + b`.
        inputs = inputs.transpose(1, 2)
        weight = self.weight[:inputs.size(2), :]
        inputs = inputs @ weight
        inputs = inputs.transpose(1, 2)
        if self.bias_enabled:
            return inputs+ self.bias
        else:
            return inputs
    def forward(self, inputs):
        # `y = xA^T + b`.
        if self.bias_enabled:
            return inputs @ self.weight.t() + self.bias
        else:
            return inputs @ self.weight.t()