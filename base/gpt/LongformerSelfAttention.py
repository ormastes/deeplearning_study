import torch
from torch import nn

from base.config.CommonConstants import CommonConstants
from base.embedding.AttentionLinearBiasPositionalEmbedding import AttentionLinearBiasPositionalEmbedding
from base.util.Log import Logger
from base.prim.Linear import Linear
import torch.nn.functional as F


class LongformerSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, head_cnt, qkv_bias=False, config=None):
        super().__init__()
        assert d_out % head_cnt == 0, "d_out must be divisible by n_heads"
        assert d_out % config.linformer_factor == 0, "d_out must be divisible by linformer_factor"
        self.config = config
        self.log = Logger.get_instance()
        self.attention_window = config.attention_window
        self.attention_dilation = config.attention_dilation
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
        assert (
                           d_out / config.linformer_factor) % self.attention_groups == 0, "d_out must be divisible by attention_groups"

        self.alibi = config.alibi(head_cnt)
        if self.config is not None and self.config.attention_window > 0:
            self.W_local_query = Linear(d_in, int(d_out / config.linformer_factor), bias=qkv_bias)
            self.W_local_key = Linear(d_in, int(d_out / config.linformer_factor) // self.attention_groups,
                                      bias=qkv_bias)
            self.W_local_value = Linear(d_in, d_out // self.attention_groups, bias=qkv_bias)

        #self.attn_weight = Linear(d_in, d_out * 3, bias=qkv_bias)
        self.W_query = Linear(d_in, int(d_out / config.linformer_factor), bias=qkv_bias)
        self.W_key = Linear(d_in, int(d_out / config.linformer_factor) // self.attention_groups, bias=qkv_bias)
        self.W_value = Linear(d_in, d_out // self.attention_groups, bias=qkv_bias)
        self.proj = Linear(d_out, d_out, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, global_attention_mask=None):

        b, token_cnt, d_in = x.size()
        if self.config.attention_window > 0 and token_cnt // self.attention_window > 0:

            q = self.W_local_query(x)
            k = self.W_local_key(x)
            k = k.repeat(1, 1, self.attention_groups)
            v = self.W_local_value(x)
            v = v.repeat(1, 1, self.attention_groups)
            self.log.debug("q_k_v shape:", q.shape)

            padding_cnt = self.attention_window - (token_cnt % self.attention_window)

            if padding_cnt > 0:
                q = F.pad(q, (0, 0, 0, padding_cnt))
                k = F.pad(k, (0, 0, 0, padding_cnt))
                v = F.pad(v, (0, 0, 0, padding_cnt))

            chunks_count = (token_cnt + padding_cnt) // self.attention_window

            # Reshape to get chunks
            local_attention_scores, local_context, _, _ = self.forward_attn_local(k, q, v, x, chunks_count, padding_cnt)

        q = self.W_query(x)
        k = self.W_key(x)
        # repeat k's dim -1 about attention_groups times
        k = k.repeat(1, 1, self.attention_groups)
        v = self.W_value(x)
        v = v.repeat(1, 1, self.attention_groups)
        self.log.debug("q_k_v shape:", q.shape)

        attention_scores, context, b, token_cnt = self.forward_attn(k, q, v, x)
        if self.config.attention_window > 0 and token_cnt // self.attention_window > 0:

            assert global_attention_mask is not None, "global_attention_mask must be provided"
            global_attention_mask = global_attention_mask.unsqueeze(2)
            context = torch.where(global_attention_mask == 1, context, local_context)

        self.log.debug("context transposed shape:", context.shape)
        result = self.proj(context)
        self.log.debug("result shape:", result.shape)

        return result

    def forward_attn_local(self, k, q, v, x, chunks_count, padding_cnt, normalizer=None):
        b, token_cnt, d_in = x.size()

        # Reshape to get chunks
        keys = q.view(b, self.head_cnt, chunks_count, self.attention_window,
                      int(self.head_out / self.config.linformer_factor))
        queries = k.view(b, self.head_cnt, chunks_count, self.attention_window,
                         int(self.head_out / self.config.linformer_factor))
        values = v.view(b, self.head_cnt, chunks_count, self.attention_window, self.head_out)
        # chunks let effect like adding head. Ant result reduce sequence.
        self.log.debug("keys shape:", keys.shape)
        self.log.debug("queries shape:", queries.shape)
        self.log.debug("values shape:", values.shape)
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attention_scores = queries @ keys.transpose(-2, -1)
        self.log.debug("q@k attention_scores shape:", attention_scores.shape)

        if normalizer is not None:
            attention_scores = normalizer(attention_scores)

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:self.attention_window, :self.attention_window]
        # Use the mask to fill attention scores
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        if self.config is not None and self.config.alibi is not None:
            alibi_attention = self.alibi(self.attention_window).flatten()
            alibi_attention = alibi_attention.view(1, self.head_cnt, 1, self.attention_window, self.attention_window)
            alibi_attention = alibi_attention.repeat(b, 1, chunks_count, 1, 1)
            # (batch, num_heads, seq_len, seq_len) = (batch, num_heads, seq_len, seq_len) + (1, num_heads, seq_len, seq_len)
            attention_scores = attention_scores + alibi_attention

        context = self.forward_attn_post(attention_scores, values, b, token_cnt)
        return attention_scores, context, b, token_cnt

    def forward_attn(self, k, q, v, x, normalizer=None):
        b, token_cnt, d_in = x.size()
        queries = q.view(b, token_cnt, self.head_cnt, int(self.head_out / self.config.linformer_factor)).transpose(1, 2)
        keys = k.view(b, token_cnt, self.head_cnt, int(self.head_out / self.config.linformer_factor)).transpose(1, 2)
        values = v.view(b, token_cnt, self.head_cnt, self.head_out).transpose(1, 2)
        self.log.debug("keys shape:", keys.shape)
        self.log.debug("queries shape:", queries.shape)
        self.log.debug("values shape:", values.shape)
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attention_scores = queries @ keys.transpose(-2, -1)
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
        context = self.forward_attn_post(attention_scores, values, b, token_cnt)
        return attention_scores, context, b, token_cnt

    def forward_attn_post(self, attention_scores, values, b, token_cnt):
        attention_scores = attention_scores / (values.shape[-1] ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # Shape: (b, num_tokens, num_heads, head_dim)
        self.log.debug("attention_weights shape:", attention_weights.shape)
        context_ = (attention_weights @ values).transpose(1, -2)
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        self.log.debug("aw@v context shape:", context_.shape)
        context_contiguous = context_.contiguous()
        if self.config.attention_window > 0 and len(context_.shape) >= 5:
            context = context_contiguous.view(b,
                                              attention_scores.shape[2], attention_scores.shape[3],
                                              self.d_out)
            context = torch.mean(context, dim=2)
            context = F.pad(context, (0, 0, 0, token_cnt - context.shape[1]))
        else:
            context = context_contiguous.view(b, token_cnt, self.d_out)
        return context

    @staticmethod
    def update_global_attention_mask(input_ids, tokenizer, special_tokens=CommonConstants.SPECIAL_TOKENS):
        global_attention_mask = torch.zeros_like(input_ids)

        # Iterate over each sequence in the batch
        for batch_idx, seq in enumerate(input_ids):
            for token in special_tokens:
                token_id = tokenizer.encode(token)[0]
                # to tensor
                token_id = torch.tensor(token_id).to(input_ids.device)
                token_positions = (seq == token_id).nonzero(as_tuple=True)[0]
                if len(token_positions) > 0:
                    global_attention_mask[batch_idx, token_positions] = 1

        return global_attention_mask
