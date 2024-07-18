import torch
import torch.nn.functional as F
from torch import nn

from base.config.CommonConstants import CommonConstants
from base.gpt.DecoderMask import DecoderMask
from base.prim.Linear import Linear
from base.util.Log import Logger


class MultiHeadAttention(nn.Module):
    def __init__(self, config, embed_dim=None, num_heads=None, drop_rate=None, qkv_bias=None, seq_first=None):
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

        return result

    def forward_attn(self, k, q, v, x, normalizer=None):
        b, seq_len, embed_dim = x.size()
        num_heads = self.num_heads
        head_out = self.head_out
        qk_out = self.qk_out

        queries = q.view(b, seq_len, num_heads, qk_out).transpose(1, 2)
        keys = k.view(b, seq_len, num_heads, qk_out).transpose(1, 2)
        values = v.view(b, seq_len, num_heads, head_out).transpose(1, 2)
        self.log.shape("queries keys", keys, [b, num_heads, seq_len, qk_out])
        self.log.shape("values", values, [b, num_heads, seq_len, head_out])

        attention_scores = self.calc_attention(queries, keys, seq_len, normalizer)

        context = self.forward_attn_post(attention_scores, values)
        return context

    def forward_attn_local(self, k, q, v, x, chunks_count, normalizer=None):
        b, seq_len, embed_dim = x.size()
        num_heads = self.num_heads
        head_out = self.head_out
        qk_out = self.qk_out
        attention_window = self.attention_window

        # Reshape to get chunks
        queries = q.view(b, num_heads, chunks_count, attention_window, qk_out)
        keys = k.view(b, num_heads, chunks_count, attention_window, qk_out)
        values = v.view(b, num_heads, seq_len, head_out)
        # chunks let effect like adding head. Then result reduce sequence.
        self.log.shape("queries keys", keys, [b, num_heads, chunks_count, attention_window, qk_out])
        self.log.shape("values", values, [b, num_heads, seq_len, head_out])

        attention_scores = self.calc_attention(queries, keys, attention_window, normalizer, chunks_count=chunks_count)

        attention_scores = attention_scores.view(b, num_heads, chunks_count, attention_window, attention_window)
        attention_scores = torch.mean(attention_scores, dim=2)
        attention_scores = F.pad(attention_scores, (0, seq_len - attention_window, 0, seq_len - attention_window))

        context = self.forward_attn_post(attention_scores, values)
        return context

    def forward_attn_post(self, attention_scores, values):
        b = values.shape[0]
        seq_len = values.shape[2]
        head_out = self.head_out
        embed_dim = self.embed_dim
        num_heads = self.num_heads

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        self.log.shape("attention_weights", attention_weights, [b, num_heads, seq_len, seq_len])

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_ = (attention_weights @ values).transpose(1, 2)
        self.log.shape("aw@v context", context_, [b, num_heads, seq_len, head_out])
        context_contiguous = context_.contiguous()
        context = context_contiguous.view(b, seq_len, embed_dim)
        self.log.shape("Transpose context", context, [b, seq_len, embed_dim])
        return context

    def calc_attention(self, queries, keys, seq_len, normalizer, chunks_count=None):
        b = queries.shape[0]
        num_heads = self.num_heads
        head_out = self.head_out
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attention_scores = queries @ keys.transpose(-2, -1)
        self.log.shape("attention_scores", attention_scores, [b, num_heads, head_out, head_out])
        if normalizer is not None:
            attention_scores = normalizer(attention_scores)

        attention_scores = self.mask(attention_scores, seq_len)

        if self.alibi is not None:
            alibi_attention_score = self.alibi(seq_len)

            attention_window = self.attention_window
            if chunks_count is not None:
                alibi_attention_score = alibi_attention_score.flatten()
                alibi_attention_score = alibi_attention_score.view(1, num_heads, 1, attention_window, attention_window)
                alibi_attention_score = alibi_attention_score.repeat(b, 1, chunks_count, 1, 1)
            attention_scores = attention_scores + alibi_attention_score

        # Normalize to 1 by multiplying by 1/sqrt(d_k)
        DIMENSION_IDX = -1
        attention_scores = attention_scores / (keys.shape[DIMENSION_IDX] ** 0.5)

        return attention_scores
    def apply_padding(self, k, q, v, seq_len):
        padding_cnt = self.attention_window - (seq_len % self.attention_window)
        if padding_cnt > 0:
            q = F.pad(q, (0, 0, 0, padding_cnt))
            k = F.pad(k, (0, 0, 0, padding_cnt))
            #v = F.pad(v, (0, 0, 0, padding_cnt))

            return k, q, v, padding_cnt
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
    num_head = 2
    d_in = inputs.shape[1]
    block_size = inputs.shape[0]
    d_out = 2
    dropout = 0.5
    torch.manual_seed(123)
    multi_headed_attention = MultiHeadAttention(d_in, d_out, block_size, dropout, num_head)
    output = multi_headed_attention(batch)
    print("Output shape:", output.shape)
    print("Output:", output)
