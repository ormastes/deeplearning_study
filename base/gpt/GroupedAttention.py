import torch
from torch import nn

from base.embedding.AttentionLinearBiasPositionalEmbedding import AttentionLinearBiasPositionalEmbedding
from base.util.Log import Logger
from base.prim.Linear import Linear


class GroupedAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, head_cnt, qkv_bias=False, config=None):
        super().__init__()
        assert d_out % head_cnt == 0, "d_out must be divisible by n_heads"
        assert d_out % config.linformer_factor == 0, "d_out must be divisible by linformer_factor"
        self.config = config
        self.log = Logger.get_instance()
        self.d_in = d_in
        self.d_out = d_out
        self.head_cnt = head_cnt
        self.block_size = context_length
        self.dropout = dropout
        self.qkv_bias = qkv_bias
        self.head_out = d_out // head_cnt
        self.attention_groups = config.attention_groups
        self.heads_per_group = head_cnt // self.attention_groups
        assert (d_out % head_cnt == 0)
        assert self.head_cnt % self.attention_groups == 0, "head_cnt must be divisible by attention_groups"
        assert self.head_out % config.linformer_factor == 0, "head_out must be divisible by linformer_factor"
        assert (d_out / config.linformer_factor) % self.attention_groups == 0, "d_out must be divisible by attention_groups"

        self.alibi = config.alibi(head_cnt)

        #self.attn_weight = Linear(d_in, d_out * 3, bias=qkv_bias)
        self.W_query = Linear(d_in, int(d_out/config.linformer_factor), bias=qkv_bias)
        self.W_key = Linear(d_in,  int(d_out/config.linformer_factor) // self.attention_groups, bias=qkv_bias)
        self.W_value = Linear(d_in, d_out // self.attention_groups, bias=qkv_bias)
        self.proj = Linear(d_out, d_out, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x):

        #self.log.debug("attn_weight shape:", self.attn_weight.weight.shape)
        #qkv = self.attn_weight(x)
        #q, k, v = qkv.chunk(3, dim=-1)
        q = self.W_query(x)
        k = self.W_key(x)
        # repeat k's dim -1 about attention_groups times
        k = k.repeat(1, 1, self.attention_groups)
        v = self.W_value(x)
        v = v.repeat(1, 1, self.attention_groups)
        self.log.debug("q_k_v shape:", q.shape)

        context = self.forward_attn(k, q, v, x)

        self.log.debug("context transposed shape:", context.shape)
        result = self.proj(context)
        self.log.debug("result shape:", result.shape)

        return result

    def forward_attn(self, k, q, v, x, normalizer=None):
        b, token_cnt, d_in = x.size()
        queries = q.view(b, token_cnt, self.head_cnt, int(self.head_out/self.config.linformer_factor)).transpose(1, 2)
        keys = k.view(b, token_cnt, self.head_cnt, int(self.head_out/self.config.linformer_factor)).transpose(1, 2)
        values = v.view(b, token_cnt, self.head_cnt, self.head_out).transpose(1, 2)
        self.log.debug("keys shape:", keys.shape)
        self.log.debug("queries shape:", queries.shape)
        self.log.debug("values shape:", values.shape)
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attention_scores = queries @ keys.transpose(2, 3)
        self.log.debug("q@k attention_scores shape:", attention_scores.shape)

        if normalizer is not None:
            attention_scores = normalizer(attention_scores)

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:token_cnt, :token_cnt]
        # Use the mask to fill attention scores
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        if self.config is not None and self.config.alibi is not None:
            # (batch, num_heads, seq_len, seq_len) = (batch, num_heads, seq_len, seq_len) + (1, num_heads, seq_len, seq_len)
            attention_scores = attention_scores + self.alibi(token_cnt)

        attention_scores = attention_scores / (keys.shape[-1] ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # Shape: (b, num_tokens, num_heads, head_dim)
        self.log.debug("attention_weights shape:", attention_weights.shape)
        context_ = (attention_weights @ values).transpose(1, 2)
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        self.log.debug("aw@v context shape:", context_.shape)
        context_contiguous = context_.contiguous()
        context = context_contiguous.view(b, token_cnt, self.d_out)  # context.reshape(b, token_cnt, self.d_out)
        return context

