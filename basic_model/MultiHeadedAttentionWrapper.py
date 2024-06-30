import torch
from torch import nn

from basic_model.SimpleAttentionV2 import SimpleAttention_v2


class MultiHeadedAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, head_cnt, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.head_out = d_out // head_cnt
        assert (d_out == self.head_out * head_cnt)
        self.block_size = block_size
        self.dropout = dropout
        self.head_cnt = head_cnt
        self.heads = nn.ModuleList([SimpleAttention_v2(d_in, self.head_out, dropout, qkv_bias) for _ in range(head_cnt)])

    def forward(self, inputs):
        return torch.cat([head(inputs) for head in self.heads], dim=-1)


if __name__ == "__main__":
    start_context = "Every effort moves you"
    tokenizer = GPT2TikTokenizer()