import torch
from torch import nn

from basic_model.SimpleLinearV1 import SimpleLinear_v1


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, head_cnt, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.block_size = block_size
        self.dropout = dropout
        self.head_cnt = head_cnt
        self.qkv_bias = qkv_bias

        self.head_out = d_out // head_cnt
        assert (d_out % head_cnt == 0)

        self.W_q = SimpleLinear_v1(d_in, d_out, bias=qkv_bias)
        self.W_k = SimpleLinear_v1(d_in, d_out, bias=qkv_bias)
        self.W_v = SimpleLinear_v1(d_in, d_out, bias=qkv_bias)
        self.out_projection = SimpleLinear_v1(d_out, d_out, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(block_size, block_size), diagonal=1))

    def forward(self, x):
        b, token_cnt, d_in = x.size()
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        keys = k.view(b, token_cnt, self.head_cnt, self.head_out).transpose(1, 2)
        queries = q.view(b, token_cnt, self.head_cnt, self.head_out).transpose(1, 2)
        values = v.view(b, token_cnt, self.head_cnt, self.head_out).transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)
        attention_scores = attention_scores / (k.shape[-1] ** 0.5)
        attention_scores.masked_fill_(self.mask.bool()[:token_cnt, :token_cnt].unsqueeze(0).unsqueeze(0), -torch.inf)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = attention_weights @ values
        context = context.transpose(1, 2).contiguous().view(b, token_cnt, self.d_out)
        return self.out_projection(context)


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
    head_cnt = 2
    d_in = inputs.shape[1]
    block_size = inputs.shape[0]
    d_out = 2
    dropout = 0.5
    torch.manual_seed(123)
    multi_headed_attention = MultiHeadedAttention(d_in, d_out, block_size, dropout, head_cnt)
    output = multi_headed_attention(batch)
    print("Output shape:", output.shape)
    print("Output:", output)