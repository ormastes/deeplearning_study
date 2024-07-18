import enum
from base.gpt.MultiHeadAttention import MultiHeadAttention
from base.gpt.TransformerBlockSequence import SimpleTransformerBlockSequence


class ModelName(enum.Enum):
    gpt2_small_124M = "openai-community/gpt2"
    gpt2_medium_355M = "openai-community/gpt2-medium"
    gpt2_large_774M = "openai-community/gpt2-large"
    gpt2_xl_1558M = "openai-community/gpt2-xl"

class GPT2_CONFIG_124M(object):
    def __init__(self,
                 vocab_size=50257, context_len=1024,
                 embed_dim=768, embed_dim_ff_dim=3072,
                 num_heads=12, num_layers=12,
                 drop_rate=0.0, qkv_bias=False):
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.embed_dim = embed_dim
        self.embed_dim_ff_dim = embed_dim_ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias
        self.trf_blocks = SimpleTransformerBlockSequence
        self.reverse_position_embedding = False
        self.num_prim_layers = None
        self.alibi = None
        self.is_feature_attention = False
        self.linformer_factor = 1.0  # linformer_factor = 1/2^k
        self.attention_groups = 1
        self.attention = MultiHeadAttention
        self.attention_window = 0
        self.attention_dilation = 1  # TODO
        self.seq_first = False
        self.tokenizer = None
        self.no_fake_quantize = False
        self.qlora_rank = 0
        self.aq_num_codebooks = 0
        self.mcq_num_codebooks = 0
        self.mcq_codebook_size = 0


model_configs = {
    ModelName.gpt2_small_124M: GPT2_CONFIG_124M(qkv_bias=True),
    ModelName.gpt2_medium_355M: GPT2_CONFIG_124M(embed_dim=1024, embed_dim_ff_dim=4096, num_heads=16,
                                                 num_layers=24, qkv_bias=True),
    ModelName.gpt2_large_774M: GPT2_CONFIG_124M(embed_dim=1280, embed_dim_ff_dim=5120, num_heads=20,
                                                num_layers=36, qkv_bias=True),
    ModelName.gpt2_xl_1558M: GPT2_CONFIG_124M(embed_dim=1600, embed_dim_ff_dim=6400, num_heads=25,
                                              num_layers=48, qkv_bias=True),
}


class GPT2_CONFIG_124M_TRAIN(GPT2_CONFIG_124M):
    def __init__(self,
                 vocab_size=50257, context_len=1024,
                 embed_dim=768, embed_dim_ff_dim=3072,
                 num_heads=12, num_layers=12,
                 drop_rate=0.1, qkv_bias=False):
        super().__init__(vocab_size, context_len, embed_dim, embed_dim_ff_dim, num_heads, num_layers, drop_rate, qkv_bias)


class OTHER_SETTINGS(object):
    def __init__(self, learning_rate=5e-4, num_epochs=10,
                 batch_size=2, weight_decay=0.1):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay

