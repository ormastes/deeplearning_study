import torch
from torch import nn

class SimpleLinear_v1(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.rand(self.in_features, self.out_features))
        self.bias_enabled = bias
        if self.bias_enabled:
            self.bias = torch.nn.Parameter(torch.rand(self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        pass
        # self.weight = torch.nn.Parameter(torch.randn(self.out_features, self.in_features))
        #nn.init.xavier_uniform_(self.weight)
        #nn.init.xavier_uniform_(self.weight)
        #nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs):
        if self.bias_enabled:
            return inputs @ self.weight + self.bias
        else:
            return inputs @ self.weight