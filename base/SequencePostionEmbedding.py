import torch
from torch import nn


from base.Embedding import Embedding


class SequencePositionEmbedding(Embedding):
    def __init__(self, max_seq_len, d_model):
        super(SequencePositionEmbedding, self).__init__(max_seq_len, d_model)
        self.max_seq_len = max_seq_len
        self.d_model = d_model


    def forward(self, x):
        _, seq_len = x.size()
        return super().forward(torch.arange(seq_len).to(x.device))