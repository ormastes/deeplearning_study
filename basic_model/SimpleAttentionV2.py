import torch
from torch import nn

from basic_model.SimpleLinearV1 import SimpleLinear_v1
from basic_model.SimpleUtil import simple_softmax


class SimpleAttention_v2(nn.Module):

    def __init__(self, input_dim, output_dim, dropout=0.5, qkv_bias=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W_q = SimpleLinear_v1(input_dim, output_dim, bias=qkv_bias)
        print("W_q weight:", self.W_q.weight)
        self.W_k = SimpleLinear_v1(input_dim, output_dim, bias=qkv_bias)
        print("W_k weight:", self.W_k.weight)
        self.W_v = SimpleLinear_v1(input_dim, output_dim, bias=qkv_bias)
        print("W_v weight:", self.W_v.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        queries = self.W_q(inputs)
        keys = self.W_k(inputs)
        values = self.W_v(inputs)
        print("QKV values:", queries, keys, values)
        attention_scores = queries @ keys.transpose(1,2)
        d_k = keys.shape[-1]
        print("Attention scores:", attention_scores)
        attention_weights = simple_softmax(attention_scores / d_k**0.5, dim=attention_scores.ndim-1)
        print("Attention weights:", attention_weights)

        mask = torch.triu(torch.ones(attention_scores.shape), diagonal=1)
        print("Mask simple:", mask)
        masked_attention = attention_weights.masked_fill(mask.bool(), -torch.inf)
        print("Masked attention weights:", masked_attention)

        # norm_masked_attention_weights = masked_attention / masked_attention.sum(dim=-1, keepdim=True)
        norm_masked_attention_weights = simple_softmax(masked_attention/d_k**0.5, dim=masked_attention.ndim-1)

        norm_masked_attention_weights = self.dropout(norm_masked_attention_weights)

        print("Normalized masked attention weights:", norm_masked_attention_weights)
        attended_context = norm_masked_attention_weights @ values

        return attended_context

    def __repr__(self):
        return f"SimpleAttention(input_dim={self.input_dim}, output_dim={self.output_dim})"

if __name__ == "__main__":
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )
    ##############################################################
    # Attention applied to a token embedding
    print("Input shape:", inputs.shape)
    d_in = inputs.shape[1]
    d_out = 2
    torch.manual_seed(123)
    attention = SimpleAttention_v2(d_in, d_out)

    output = attention(inputs)
    print("Output shape:", output.shape)
    print("Output:", output)

