import torch
import torch.nn.functional as F
from torch import nn

from base.config.CommonConstants import CommonConstants
from base.gpt.DecoderMask import DecoderMask
from base.prim.Linear import Linear
from base.util.Log import Logger

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

class FlashAttention(nn.Module):
    def __init__(self, embed_dim=None, num_heads=None, drop_rate=None, qkv_bias=None, seq_first=None, config=None):
        super().__init__()
        self.config = config
        self.log = Logger.get_instance()
        embed_dim = config.embed_dim if embed_dim is None else embed_dim
        num_heads = config.num_heads if num_heads is None else num_heads
        seq_first = config.seq_first if seq_first is None else seq_first
        qkv_bias = config.qkv_bias if qkv_bias is None else qkv_bias
        drop_rate = config.drop_rate if drop_rate is None else drop_rate
        head_out = embed_dim // num_heads
        linformer_factor = config.linformer_factor
        context_len = config.context_len
        attention_groups = config.attention_groups
        attention_window = config.attention_window
        attention_dilation = config.attention_dilation  # not used yet

        qk_embed_dim = embed_dim
        qk_out = head_out

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_first = seq_first
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.head_out = head_out
        self.qk_embed_dim = qk_embed_dim
        self.qk_out = qk_out

        self.linformer_factor = linformer_factor
        if linformer_factor > 1.0:
            assert head_out % linformer_factor == 0, "head_out must be divisible by linformer_factor"
            qk_embed_dim = int(qk_embed_dim / linformer_factor)
            qk_out = int(qk_out / linformer_factor)

            self.qk_embed_dim = qk_embed_dim
            self.qk_out = qk_out

        self.attention_groups = attention_groups
        if attention_groups != 0:
            heads_per_group = num_heads // self.attention_groups
            assert qk_embed_dim % attention_groups == 0, "head_cnt must be divisible by attention_groups"
            assert qk_out % attention_groups == 0, "key_query_out must be divisible by attention_groups"
            qk_embed_dim = int(qk_embed_dim / attention_groups)
            qk_out = int(qk_out / attention_groups)

            self.heads_per_group = heads_per_group
            self.qk_embed_dim = qk_embed_dim
            self.qk_out = qk_out

        self.attention_window = attention_window
        if attention_window > 0:
            self.attention_dilation = attention_dilation  # not used yet
            self.W_local_query = Linear(embed_dim, qk_embed_dim, bias=qkv_bias)
            self.W_local_key = Linear(embed_dim, qk_embed_dim, bias=qkv_bias)
            self.W_local_value = Linear(embed_dim, embed_dim, bias=qkv_bias)

        if config.alibi is not None:
            self.alibi = config.alibi(num_heads)
        else:
            self.alibi = None

        self.W_query = Linear(embed_dim, qk_embed_dim, bias=qkv_bias)
        self.W_key = Linear(embed_dim, qk_embed_dim, bias=qkv_bias)
        self.W_value = Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj = Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(drop_rate)
        self.mask = DecoderMask(context_len)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            prefix+'W_query': self.W_query.state_dict(),
            prefix+'W_key': self.W_key.state_dict(),
            prefix+'W_value': self.W_value.state_dict(),
            prefix+'proj': self.proj.state_dict(),
            prefix+'dropout': self.dropout.state_dict()
        }
        if self.attention_window > 0:
            state_dict[prefix+'W_local_query'] = self.W_local_query.state_dict()
            state_dict[prefix+'W_local_key'] = self.W_local_key.state_dict()
            state_dict[prefix+'W_local_value'] = self.W_local_value.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.W_query.load_state_dict(state_dict['W_query'])
        self.W_key.load_state_dict(state_dict['W_key'])
        self.W_value.load_state_dict(state_dict['W_value'])
        self.proj.load_state_dict(state_dict['proj'])
        self.dropout.load_state_dict(state_dict['dropout'])
        if self.attention_window > 0:
            self.W_local_query.load_state_dict(state_dict['W_local_query'])
            self.W_local_key.load_state_dict(state_dict['W_local_key'])
            self.W_local_value.load_state_dict(state_dict['W_local_value'])

    def forward(self, x, global_attention_mask=None):
        qk_embed_dim = self.qk_embed_dim
        shape_len = len(x.shape)
        if shape_len > 3:
            # main tain batch and squence only
            original_shape = x.shape
            x = x.view(x.shape[0], x.shape[1], -1)
            x = x.transpose(1, 2)
        b, seq_len, embed_dim = x.size()

        if self.attention_window > 0:
            q = self.W_local_query(x)
            k = self.W_local_key(x)
            v = self.W_local_value(x)

            self.log.shape("weight applied queries keys", k, [b, seq_len, qk_embed_dim])
            self.log.shape("weight applied values", v, [b, seq_len, embed_dim])

            q, k, v, padding_cnt= self.apply_padding(q, k, v, seq_len)

            chunks_count = (seq_len + padding_cnt) // self.attention_window

            # Reshape to get chunks
            local_context = self.forward_attn_local(k, q, v, x, chunks_count)

        q = self.W_query(x)
        k = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        v = self.W_value(x)
        self.log.shape("weight applied queries keys", k, [b, seq_len, qk_embed_dim])
        self.log.shape("weight applied values", v, [b, seq_len, embed_dim])

        context = self.forward_attn(k, q, v, x)

        if self.config.attention_window > 0:
            assert global_attention_mask is not None, "global_attention_mask must be provided"
            assert local_context is not None, "local_context must be provided"
            global_attention_mask = global_attention_mask.unsqueeze(2)
            context = torch.where(global_attention_mask == 1, context, local_context)

        result = self.proj(context)
        self.log.shape("result context", result, [b, seq_len, embed_dim])
        if shape_len > 3:
            result = result.transpose(1, 2)
            result = result.view(original_shape)
        return result

    def forward_attn(self, k, q, v, x):
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_out)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_out)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_out)

        # alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
        #             (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
        #             is added to the attention score of query i and key j.
        alibi = None
        if self.alibi is not None:
            alibi = torch.Tensor([1.0 for i in range(self.num_heads)]).to(self.config.device) # self.alibi(k.size(1))

        context = flash_attn_func(q, k, v, alibi_slopes=alibi)
        context = context.view(context.size(0), context.size(1), -1)
        return context
        # softmax_scale=None, causal=False, window_size=(-1, -1), alibi_slopes=alibi, deterministic=False)