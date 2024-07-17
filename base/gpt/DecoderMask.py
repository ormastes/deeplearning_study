import torch


class DecoderMask(torch.nn.Module):
    def __init__(self, context_len):
        super().__init__()
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, attention_scores, seq_len):
        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        # Use the mask to fill attention scores
        attention_scores.masked_fill_(mask_bool, -torch.inf)
        return  attention_scores.masked_fill_(mask_bool, -torch.inf)