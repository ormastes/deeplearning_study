import torch
from base.gpt.TransformerBlock import TransformerBlock
from base.util.Log import Logger
from base.prim.LayerNorm import LayerNorm


class SimpleTransformerBlockSequence(torch.nn.Sequential):
    def __init__(self, config):
        super().__init__(*[TransformerBlock(config) for _ in range(config.num_layers)])

    def forward(self, x, local_attention_scores=None):
        for block in self:
            x = block(x, local_attention_scores)
        return x


class SharedTransformerBlockSequence(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = Logger.get_instance()
        self.blocks = torch.nn.ModuleList([TransformerBlock(config) for _ in range(config.num_prim_layers)])
        self.norms = torch.nn.ModuleList([LayerNorm(config.embed_dim) for _ in range(config.num_layers)])

    def forward(self, x):
        block_idx = 0
        for qb in self.blocks:
            for kb in self.blocks:
                for vb in self.blocks:

                    self.log.info("Block input shape:", x.shape)
                    shortcut = x
                    if qb.front_norm :
                        x = qb.norm1(x)

                    q = qb.attn.W_query(x)
                    k = kb.attn.W_key(x)  # Shape: (b, num_tokens, d_out)
                    v = vb.attn.W_value(x)
                    context = vb.attn.forward_attn(k, q, v, x, self.norms[block_idx])
                    x = vb.attn.proj(context)

                    x = vb.forward_after_attn(x, shortcut)

                    block_idx += 1

        return x
