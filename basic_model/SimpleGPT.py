import torch
from torch import nn

from basic_model.BPETokenizer import GPT2TikTokenizer
from basic_model.MultiHeadedAttention import MultiHeadedAttention
from basic_model.SimpleEmbedding import SimpleEmbedding
from basic_model.SimpleGPTConfig import GPT2_CONFIG_124M
from basic_model.SimpleLinearV1 import SimpleLinear_v1
from basic_model.SequencePostionEmbedding import SequencePositionEmbedding

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((x * (1 + 0.044715 * x * x))))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential(
            SimpleLinear_v1(config.embed_dim, config.embed_dim_ff_dim),
            GELU(),
            nn.Dropout(config.drop_rate),
            SimpleLinear_v1(config.embed_dim_ff_dim, config.embed_dim),
            nn.Dropout(config.drop_rate)
        )
    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.attn = MultiHeadedAttention(config.embed_dim, config.embed_dim, config.ctx_len,
                                         config.drop_rate, config.num_heads, config.qkv_bias)

        self.ff = FeedForward(config)

        self.ln1 = LayerNorm(config.embed_dim)
        self.ln2 = LayerNorm(config.embed_dim)

        self.drop = nn.Dropout(config.drop_rate)

    def forward(self, x):
        shortcut = x

        x = self.ln1(x + self.drop(self.attn(x)))
        x = self.attn(x)
        x = self.drop(x)
        x = x+shortcut

        x = self.ln(x)
        x = self.ff(x)
        x = self.drop(x)
        x = x+shortcut
        return x
    def forward(self, x):
        return x

class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        # weight has both scale and shift
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False to use the same normalization as GPT-2
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / (var + self.eps).sqrt()
        return norm_x * self.scale + self.shift



class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embed = SimpleEmbedding(config.vocab_size, config.embed_dim)  # embeddings does not have size
        self.pos_embed = SequencePositionEmbedding(config.ctx_len, config.embed_dim)
        self.drop = nn.Dropout(config.drop_rate)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.num_layers)])  # list to elements
        self.final_layer_norm = LayerNorm(config.embed_dim)
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


def generate_text_simple(model, idx, max_new_tokens, context_len):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_len:]

        logits = model(idx_cond)
        probas = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat([idx, idx_next], dim=-1)

        next_token = torch.argmax(logits, dim=-1)
        idx = torch.cat([idx, next_token], dim=-1)
    return idx


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
    model = GPTModel(config)
    logits = model(batch)
    print("Logits shape:", logits.shape)
    print("Logits:", logits)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total params:", total_params)

    ##############################################################
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    transformer_block = TransformerBlock(config)
    output = transformer_block(x)
    print("Output shape:", output.shape)

    # for name, param in model.named_parameters():
    #    print(name, "\t", param.numel(),"\t", param.shape)
    #    #print(param)

    ##############################################################


    start_context = "Hello, I am"
    tokenizer = GPT2TikTokenizer()
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("Encoded shape:", encoded_tensor.shape)
    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_len=config.ctx_len
    )
    print("Output shape:", out.shape)
    print("Output:", out)
    print("Output text:", tokenizer.decode(out.squeeze(0).tolist()))
