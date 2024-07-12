from torch import nn

from base.prim.FeedForward import FeedForward
from base.prim.LayerNorm import LayerNorm
from base.gpt.MultiHeadAttention import MultiHeadAttention
from base.util.Log import Logger


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = Logger.get_instance()
        self.config = config

        self.norm1 = LayerNorm(config.embed_dim)

        self.attn = MultiHeadAttention(config.embed_dim, config.embed_dim, config.context_length,
                                         config.drop_rate, config.num_heads, config.qkv_bias)

        self.norm2 = LayerNorm(config.embed_dim)

        # feed forward, mlp = multi-layer perceptron
        self.mlp = FeedForward(config)

        self.drop = nn.Dropout(config.drop_rate)
        self.front_norm = True

    def forward(self, x):
        self.log.info("Block input shape:", x.shape)
        shortcut = x
        if self.front_norm :
            x = self.norm1(x)
        x = self.attn(x)  # Shape [batch_size, num_tokens, emb_size]
        self.log.info("Block Attention output shape:", x.shape)
        x = self.drop(x)
        x = x + shortcut
        if not self.front_norm:
            x = self.norm1(x)

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        self.log.info("Block FF output shape:", x.shape)
        x = self.drop(x)
        x = x+shortcut
        self.log.info("Block output shape:", x.shape)
        return x
