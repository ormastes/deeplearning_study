import torch
from torch import nn
import math
from base.embedding.AttentionLinearBiasPositionalEmbedding import AttentionLinearBiasPositionalEmbedding
from base.util.Log import Logger
from base.prim.Linear import Linear

class FeatureAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, head_cnt, qkv_bias=False, config=None, token_cnt_limit=512):
        super().__init__()
        assert d_out % head_cnt == 0, "d_out must be divisible by n_heads"
        self.config = config
        self.log = Logger.get_instance()
        self.d_in = d_in
        self.d_out = d_out
        self.head_cnt = head_cnt
        self.block_size = context_length
        self.dropout = dropout
        self.qkv_bias = qkv_bias
        self.head_out = d_out // head_cnt
        assert (d_out % head_cnt == 0)

        self.alibi = config.alibi(head_cnt)
        feature_alibi = torch.tensor([1/i for i in range(context_length,0,-1)]).float()
        self.feature_alibi = torch.nn.Parameter(feature_alibi, requires_grad=False).view(1, 1, context_length, 1)

        # self.attn_weight = Linear(d_in, d_out * 3, bias=qkv_bias)
        self.W_query = Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = Linear(d_in, d_out, bias=qkv_bias)
        self.proj = Linear(d_out, d_out, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):

        # self.log.debug("attn_weight shape:", self.attn_weight.weight.shape)
        # qkv = self.attn_weight(x)
        # q, k, v = qkv.chunk(3, dim=-1)
        q = self.W_query(x)
        k = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        v = self.W_value(x)
        self.log.debug("q_k_v shape:", q.shape)

        context = self.forward_attn(k, q, v, x)

        self.log.debug("context transposed shape:", context.shape)
        result = self.proj(context)
        self.log.debug("result shape:", result.shape)

        return result

    def forward_attn(self, k, q, v, x, normalizer=None):
        b, token_cnt, d_in = x.size()
        queries = q.view(b, token_cnt, self.head_cnt, self.head_out).transpose(1, 2)
        keys = k.view(b, token_cnt, self.head_cnt, self.head_out).transpose(1, 2)
        values = v.view(b, token_cnt, self.head_cnt, self.head_out).transpose(1, 2)
        self.log.debug("keys shape:", keys.shape)
        self.log.debug("queries shape:", queries.shape)
        self.log.debug("values shape:", values.shape)
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # >> Feature attention different point
        attention_scores = queries.transpose(2, 3) @ keys
        self.log.debug("q@k attention_scores shape:", attention_scores.shape)

        if normalizer is not None:
            attention_scores = normalizer(attention_scores)

        # >> Feature attention different point
        if self.config.is_feature_attention == False:
            # Original mask truncated to the number of tokens and converted to boolean
            mask_bool = self.mask.bool()[:token_cnt, :token_cnt]
            # Use the mask to fill attention scores
            attention_scores.masked_fill_(mask_bool, -torch.inf)
        # >> Feature attention different point
        if False and self.config is not None and self.config.alibi is not None:
            # (batch, num_heads, seq_len, seq_len) = (batch, num_heads, seq_len, seq_len) + (1, num_heads, seq_len, seq_len)
            attention_scores = attention_scores + self.alibi(d_in)

        attention_scores = attention_scores / (keys.shape[-1] ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # Shape: (b, num_tokens, num_heads, head_dim)
        self.log.debug("attention_weights shape:", attention_weights.shape)
        # >> Feature attention different point
        feature_alibi = self.feature_alibi[:, :, -token_cnt:, :].to(values.device)
        values = values + feature_alibi
        # >> Feature attention different point
        context_ = (attention_weights @ values.transpose(2, 3)).transpose(2, 3).transpose(1, 2)
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        self.log.debug("aw@v context shape:", context_.shape)
        context_contiguous = context_.contiguous()
        context = context_contiguous.view(b, token_cnt, self.d_out)  # context.reshape(b, token_cnt, self.d_out)
        return context