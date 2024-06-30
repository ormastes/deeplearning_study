import torch
from torch import nn
from base.Linear import Linear

class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.block_size = block_size
        self.W_q = Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(block_size, block_size), diagonal=1))
    def forward(self, x):
        b, token_cnt, d_in = x.size()
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        attention_scores = q @ k.transpose(1, 2)
        attention_scores.masked_fill_(self.mask.bool()[:token_cnt, :token_cnt], -torch.inf)
        attention_weights = torch.nn.functional.softmax(attention_scores / (k.shape[-1] ** 0.5), dim=1)
        attention_weights = self.dropout(attention_weights)
        return attention_weights @ v

if __name__ == "__main__":
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )
    batch = torch.stack((inputs, inputs), dim=0)
    print("Batch shape:", batch.shape)
    d_in = inputs.shape[1]
    d_out = 2
    block_size = inputs.shape[0]
    dropout = 0.5
    torch.manual_seed(123)
    casual_attention = CasualAttention(d_in, d_out, block_size, dropout)
    output = casual_attention(batch)
    print("Output shape:", output.shape)