import torch
import math
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