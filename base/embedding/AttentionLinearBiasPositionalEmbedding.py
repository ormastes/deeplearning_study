import torch
import math
from torch.distributions import Normal
from base.prim.Linear import Linear
class AttentionLinearBiasPositionalEmbedding(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        # Create slopes for each head, scaling linearly
        slopes = torch.tensor([(i + 1) for i in range(num_heads)]).float()
        self.slopes = torch.nn.Parameter(slopes, requires_grad=False)

    def forward(self, seq_len):
        # Create a bias matrix for the sequence
        # next makes a 2D tensor of shape (seq_len, seq_len)
        # [[0, 1, 2, 3, 4],
        #  [-1, 0, 1, 2, 3],
        #  [-2, -1, 0, 1, 2],
        #  [-3, -2, -1, 0, 1],
        #  [-4, -3, -2, -1, 0]]
        alibi_bias = torch.arange(seq_len).to(self.slopes.device).unsqueeze(0) - torch.arange(seq_len).to(self.slopes.device).unsqueeze(1)

        alibi_bias = alibi_bias.float().unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)
        slopes_view = self.slopes.view(1, self.num_heads, 1, 1) # (1, num_heads, 1, 1)
        # Each head has a different slope weight
        alibi_bias = alibi_bias * slopes_view # (1, num_heads, seq_len, seq_len)
        return alibi_bias


class SimpleLearnableAlibiPositionalEmbedding(torch.nn.Module):
    @staticmethod
    def zero_grad_hook(grad):
        grad.mul_(0.1)  # break point does not work
        return grad

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        # Create slopes for each head, scaling linearly
        slopes = torch.tensor([(i + 1) for i in range(num_heads)]).float()
        self.slopes = torch.nn.Parameter(slopes, requires_grad=False)
        # initial value of b is 1
        self.b = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b.register_hook(self.zero_grad_hook)

    def forward(self, seq_len):
        if seq_len > 1:
            dynamic = torch.randn(1) * (1.1 - 0.9) + 0.9  # Scale to [0.9, 1.1]
        else:
            dynamic = torch.ones(1)

        dynamic_b = dynamic.to(self.b.device) * self.b

        alibi_bias_row = torch.arange(seq_len).to(self.slopes.device)
        alibi_bias_row = alibi_bias_row.unsqueeze(0)
        alibi_bias_col = torch.arange(seq_len).to(self.slopes.device)
        alibi_bias_col = alibi_bias_col.unsqueeze(1)
        alibi_bias = alibi_bias_row - alibi_bias_col

        alibi_bias = alibi_bias * dynamic_b

        alibi_bias = alibi_bias.float().unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)
        slopes_view = self.slopes.view(1, self.num_heads, 1, 1) # (1, num_heads, 1, 1)
        # Each head has a different slope weight
        alibi_bias = alibi_bias * slopes_view # (1, num_heads, seq_len, seq_len)
        return alibi_bias



class ComplexLearnableAlibiPositionalEmbedding(torch.nn.Module):
    @staticmethod
    def zero_grad_hook(grad):
        grad.mul_(1.0)  # break point does not work
        return grad

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        # Create slopes for each head, scaling linearly
        slopes = torch.tensor([(i + 1) for i in range(num_heads)]).float()
        self.slopes = torch.nn.Parameter(slopes, requires_grad=False)
        # initial value of b is 1
        # Learnable node count
        self.learnable_node_count = 32
        self.a = torch.nn.Parameter(torch.tensor([1.0 if i==0 else 0.1 for i in range(self.learnable_node_count)]).float().unsqueeze(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.tensor([(i) for i in range(self.learnable_node_count)]).float().unsqueeze(1), requires_grad=True)

    def forward(self, seq_len):
        if seq_len > 1:
            dynamic = torch.randn(1) * (1.1 - 0.9) + 0.9  # Scale to [0.9, 1.1]
        else:
            dynamic = torch.ones(1)

        alibi_bias_ = dynamic.to(self.slopes.device) * torch.arange(seq_len).to(self.slopes.device)
        alibi_bias_ = torch.nn.functional.relu(alibi_bias_.unsqueeze(0)*self.a.to(self.slopes.device)+ self.b.to(self.slopes.device))
        # sum first dimension to get the 1D tensor
        alibi_bias_ = alibi_bias_.sum(dim=0)
        alibi_bias_row = alibi_bias_
        alibi_bias_row = alibi_bias_row.unsqueeze(0)
        alibi_bias_col = alibi_bias_
        alibi_bias_col = alibi_bias_col.unsqueeze(1)
        alibi_bias = alibi_bias_row - alibi_bias_col

        alibi_bias = alibi_bias.float().unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)
        slopes_view = self.slopes.view(1, self.num_heads, 1, 1) # (1, num_heads, 1, 1)
        # Each head has a different slope weight
        alibi_bias = alibi_bias * slopes_view # (1, num_heads, seq_len, seq_len)
        return alibi_bias