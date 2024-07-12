import torch
import math


class SequencePositionEmbedding(torch.nn.Module):
    def __init__(self, max_seq_len, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.weight = torch.zeros(max_seq_len, vocab_size, requires_grad=False)

        # Generate position indices
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) # position: [max_seq_len, 1]

        # Generate the division term (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, vocab_size, 2).float() * (-math.log(10000.0) / vocab_size)) # div_term: [vocab_size/2, 1]

        # exp applied to position
        exp_position = position * div_term

        # Apply sin to even indices and cos to odd indices
        self.weight[:, 0::2] = torch.sin(exp_position)
        self.weight[:, 1::2] = torch.cos(exp_position)

        # Add a batch dimension
        self._parameters = {"weights": self.weight}


    def forward(self, x):
        _, seq_len, _ = x.size()
        return self.weight[:seq_len, :]
