import torch
import torch.nn as nn
import math
from base.quantization.QuantizedLinear import QuantizedLinear


class QuantizedAttention(nn.Module):
    def __init__(self, config, embed_dim=None, num_heads=None, drop_rate=None, seq_first=None):
        super(QuantizedAttention, self).__init__()
        self.config = config
        embed_dim = config.embed_dim if embed_dim is None else embed_dim
        num_heads = config.num_heads if num_heads is None else num_heads
        seq_first = config.seq_first if seq_first is None else seq_first
        drop_rate = config.drop_rate if drop_rate is None else drop_rate
        head_out = embed_dim // num_heads
        no_fake_quantize = config.no_fake_quantize

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_first = seq_first
        self.head_dim = head_out

        self.W_query = QuantizedLinear(embed_dim, embed_dim, bias=config.qkv_bias, no_fake_quantize=no_fake_quantize)
        self.W_key = QuantizedLinear(embed_dim, embed_dim, bias=config.qkv_bias, no_fake_quantize=no_fake_quantize)
        self.W_value = QuantizedLinear(embed_dim, embed_dim, bias=config.qkv_bias, no_fake_quantize=no_fake_quantize)
        self.proj = QuantizedLinear(embed_dim, embed_dim, no_fake_quantize=no_fake_quantize)
        self.dropout = nn.Dropout(drop_rate)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            prefix+'W_query': self.W_query.state_dict(),
            prefix+'W_key': self.W_key.state_dict(),
            prefix+'W_value': self.W_value.state_dict(),
            prefix+'proj': self.proj.state_dict(),
            prefix+'dropout': self.dropout.state_dict()
        }
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.W_query.load_state_dict(state_dict['W_query'])
        self.W_key.load_state_dict(state_dict['W_key'])
        self.W_value.load_state_dict(state_dict['W_value'])
        self.proj.load_state_dict(state_dict['proj'])
        self.dropout.load_state_dict(state_dict['dropout'])
    def forward(self, x, global_attention_mask=None):
        if not self.seq_first:
            x = x.transpose(0, 1)

        # Linear projections
        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        # Attention mechanism
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        # Final linear projection
        output = self.proj(attn_output)

        if not self.seq_first:
            output = output.transpose(0, 1)

        return output
