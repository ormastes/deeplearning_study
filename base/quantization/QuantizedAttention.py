import torch
import torch.nn as nn
import math
from base.quantization.QuantizedLinear import QuantizedLinear


class QuantizedAttention(nn.Module):
    def __init__(self, config, embed_dim=None, num_heads=None, dropout=None, seq_first=None):
        super(QuantizedAttention, self).__init__()
        self.config = config
        embed_dim = config.embed_dim if embed_dim is None else embed_dim
        num_heads = config.num_heads if num_heads is None else num_heads
        seq_first = config.seq_first if seq_first is None else seq_first

        assert embed_dim % num_heads == 0, "d_out must be divisible by n_heads"
        assert embed_dim % config.linformer_factor == 0, "d_out must be divisible by linformer_factor"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_first = seq_first

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.W_query = QuantizedLinear(embed_dim, embed_dim)
        self.W_key = QuantizedLinear(embed_dim, embed_dim)
        self.W_value = QuantizedLinear(embed_dim, embed_dim)
        self.proj = QuantizedLinear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        if not self.seq_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Apply fake quantization to input tensors
        query = self.W_query.fake_quant(query)
        key = self.W_key.fake_quant(key)
        value = self.W_value.fake_quant(value)

        # Linear projections
        q = self.W_query(query)
        k = self.W_key(key)
        v = self.W_value(value)

        # Attention mechanism
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        # Apply fake quantization to output
        attn_output = self.proj.fake_quant(attn_output)

        # Final linear projection
        output = self.proj(attn_output)

        if not self.seq_first:
            output = output.transpose(0, 1)

        return output