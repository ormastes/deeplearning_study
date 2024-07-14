import torch
from torch import nn

from base.embedding.AttentionLinearBiasPositionalEmbedding import AttentionLinearBiasPositionalEmbedding
from base.util.Log import Logger
from base.prim.Linear import Linear


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, head_cnt, qkv_bias=False, config=None):
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

        #self.attn_weight = Linear(d_in, d_out * 3, bias=qkv_bias)
        self.W_query = Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = Linear(d_in, d_out, bias=qkv_bias)
        self.proj = Linear(d_out, d_out, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x):

        #self.log.debug("attn_weight shape:", self.attn_weight.weight.shape)
        #qkv = self.attn_weight(x)
        #q, k, v = qkv.chunk(3, dim=-1)
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
        context = (attention_weights @ values).transpose(1, 2)
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        self.log.debug("aw@v context shape:", context.shape)
        context = context.contiguous().view(b, token_cnt, self.d_out)  # context.reshape(b, token_cnt, self.d_out)
        return context


if __name__ == "__main__":
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )
    batch = torch.stack((inputs, inputs), dim=0)
    print("Batch shape:", batch.shape)
    head_cnt = 2
    d_in = inputs.shape[1]
    block_size = inputs.shape[0]
    d_out = 2
    dropout = 0.5
    torch.manual_seed(123)
    multi_headed_attention = MultiHeadAttention(d_in, d_out, block_size, dropout, head_cnt)
    output = multi_headed_attention(batch)
    print("Output shape:", output.shape)
    print("Output:", output)