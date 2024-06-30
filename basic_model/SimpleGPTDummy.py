import torch
from torch import nn

from basic_model.BPETokenizer import GPT2TikTokenizer
from basic_model.SimpleEmbedding import SimpleEmbedding
from basic_model.SimpleGPTConfig import GPT2_CONFIG_124M
from basic_model.SimpleLinearV1 import SimpleLinear_v1
from basic_model.SequencePostionEmbedding import SequencePositionEmbedding


class DummyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def forward(self, x):
        return x

class LayerNorm(nn.Module):
    SCALE = 0
    SHIFT = 1
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        # weight has both scale and shift
        self.weight = nn.Parameter(torch.tensor([torch.ones(embed_dim), torch.zeros(embed_dim)]))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False to use the same normalization as GPT-2
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / (var + self.eps).sqrt()
        return norm_x * self.weight[LayerNorm.SCALE] + self.weight[LayerNorm.SHIFT]

class DummyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embed = SimpleEmbedding(config.vocab_size, config.embed_dim)
        self.pos_embed = SequencePositionEmbedding(config.ctx_len, config.embed_dim)
        self.drop = nn.Dropout(config.drop_rate)
        self.blocks = nn.Sequential(*[DummyBlock(config) for _ in range(config.num_layers)])  # list to elements
        self.final_layer_norm = DummyLayerNorm(config.embed_dim)
        self.head = SimpleLinear_v1(config.embed_dim, config.vocab_size, bias=False)
    def forward(self, x):
        batch_size, seq_len = x.size()
        token_embed = self.tok_embed(x)
        pos = self.pos_embed(x)
        x = self.drop(token_embed + pos)
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.head(x)
        return logits


if __name__ == "__main__":
    config = GPT2_CONFIG_124M()
    tokenizer = GPT2TikTokenizer()
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch =torch.stack(batch, dim=0)
    print("Batch shape:", batch.shape)

    torch.manual_seed(123)
    model = DummyGPTModel(config)
    logits = model(batch)
    print("Logits shape:", logits.shape)
    print("Logits:", logits)