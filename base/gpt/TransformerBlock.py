from torch import nn
from base.gpt.FeatureAttention import FeatureAttention
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

        self.attn = config.attention(config=config)
        if self.config.is_feature_attention:
            self.feature_attn = FeatureAttention(config.embed_dim, config.embed_dim, config.context_len,
                                       config.drop_rate, config.num_heads, config.qkv_bias, config=config)

        self.norm2 = LayerNorm(config.embed_dim)

        # feed forward, mlp = multi-layer perceptron
        self.mlp = FeedForward(config)

        self.drop = nn.Dropout(config.drop_rate)
        self.front_norm = True

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            prefix+'norm1': self.norm1.state_dict(),
            prefix+'attn': self.attn.state_dict(),
            prefix+'norm2': self.norm2.state_dict(),
            prefix+'mlp': self.mlp.state_dict(),
            prefix+'drop': self.drop.state_dict()
        }
        if self.config.is_feature_attention:
            state_dict['feature_attn'] = self.feature_attn.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.norm1.load_state_dict(state_dict['norm1'])
        self.attn.load_state_dict(state_dict['attn'])
        self.norm2.load_state_dict(state_dict['norm2'])
        self.mlp.load_state_dict(state_dict['mlp'])
        self.drop.load_state_dict(state_dict['drop'])
        if self.config.is_feature_attention:
            self.feature_attn.load_state_dict(state_dict['feature_attn'])


    def forward(self, x, local_attention_scores=None):
        self.log.shape("Block input", x, x.shape)
        shortcut = x
        if self.front_norm :
            x = self.norm1(x)
        if self.config.attention_window > 0:
            x = self.attn(x, local_attention_scores)
        else:
            x = self.attn(x)

        if self.config.is_feature_attention:
            x_pre = x
            x = self.feature_attn(x)
            x = x_pre*0.9 + x*0.1

        self.log.shape("Block Attention output", x, x.shape)

        return self.forward_after_attn(x, shortcut)

    def forward_after_attn(self, x, shortcut):
        x = self.drop(x)
        x = x + shortcut
        if not self.front_norm:
            x = self.norm1(x)
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        self.log.shape("Block FF output", x, x.shape)
        x = self.drop(x)
        x = x + shortcut
        self.log.shape("Block output", x, x.shape)
        return x
