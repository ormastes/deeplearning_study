import torch

# embedding is an efficient way to implement one-hot encoding by using a matrix multiplication.

# implement embedding using a simple matrix multiplication with embedding itself.
class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weight = torch.randn(vocab_size, embedding_size, requires_grad=True)
        self._parameters = {"weights": self.weight}

    def forward(self, input_ids):
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=self.vocab_size).to(self.weight.dtype)
        if one_hot.device != self.weight.device:
            one_hot = one_hot.to(self.weight.device)
        return one_hot @ self.weight

    def __repr__(self):
        return f"SimpleEmbedding(vocab_size={self.vocab_size}, output_dim={self.embedding_size})"




