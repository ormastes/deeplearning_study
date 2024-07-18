import torch
import math


class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, max_seq_len, embedding_size, config):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.weight = torch.zeros(max_seq_len, embedding_size, requires_grad=False)
        self.is_reverse = config.reverse_position_embedding

        # Generate position indices
        if self.is_reverse:
            position = torch.arange(max_seq_len-1, 0-1, -1, dtype=torch.float).unsqueeze(1)
        else:
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # position: [max_seq_len, 1]

        # Generate the division term (10000^(2i/embedding_size)) -> exp(2i*(log(10000)/embedding_size))
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))  # div_term: [vocab_size/2]

        # exp applied to position
        exp_position = position * div_term  # exp_position: [max_seq_len, vocab_size/2]

        # Apply sin to even indices and cos to odd indices
        self.weight[:, 0::2] = torch.sin(exp_position)
        self.weight[:, 1::2] = torch.cos(exp_position)


        # Add a batch dimension
        self._parameters = {"weights": self.weight}

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'weight': self.weight
        }
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.weight = state_dict['weight']

    def forward(self, x):
        _, seq_len, _ = x.size()
        if self.is_reverse:
            return self.weight[-seq_len:, :]
        else:
            return self.weight[:seq_len, :]
