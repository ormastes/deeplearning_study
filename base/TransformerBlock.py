import torch
from torch import nn

from base.FeedForward import FeedForward
from base.LayerNorm import LayerNorm
from base.MultiHeadAttention import MultiHeadAttention
from base.Log import Logger


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = Logger.get_instance()
        self.config = config

        self.ln_1 = LayerNorm(config.embed_dim)

        self.attn = MultiHeadAttention(config.embed_dim, config.embed_dim, config.context_length,
                                         config.drop_rate, config.num_heads, config.qkv_bias)

        self.ln_2 = LayerNorm(config.embed_dim)

        # feed forward, mlp = multi-layer perceptron
        self.mlp = FeedForward(config)

        self.drop = nn.Dropout(config.drop_rate)

    def forward(self, x):
        self.log.info("Block input shape:", x.shape)
        shortcut = x

        x = self.ln_1(x + self.drop(self.attn(x)))

        self.log.info("Block Attention output shape:", x.shape)
        x = self.drop(x)
        x = x+shortcut

        x = self.ln_2(x)
        x = self.mlp(x)
        self.log.info("Block FF output shape:", x.shape)
        x = self.drop(x)
        x = x+shortcut
        self.log.info("Block output shape:", x.shape)
        return x