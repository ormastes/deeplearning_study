import torch
from torch import nn


from base.Embedding import Embedding


class SequencePositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super(SequencePositionEmbedding, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embedding = Embedding(max_seq_len, d_model)

    def forward(self, x):
        _, seq_len = x.size()
        return self.embedding(torch.arange(seq_len).to(x.device))