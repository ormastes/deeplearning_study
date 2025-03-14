import torch
from base.dataset.SimpleDataset import create_data_loader

# embedding is an efficient way to implement one-hot encoding by using a matrix multiplication.

# implement embedding using a simple matrix multiplication with embedding itself.
class SimpleEmbedding:
    def __init__(self, vocab_size, output_dim):
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.weight = torch.randn(vocab_size, output_dim, requires_grad=True)

    def __call__(self, input_ids):
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=self.vocab_size).to(self.weight.dtype)
        return one_hot @ self.weight

    def __repr__(self):
        return f"SimpleEmbedding(vocab_size={self.vocab_size}, output_dim={self.output_dim})"

class SimpleGPT2Embedding:
    def __init__(self, vocab_size, output_dim, block_size):
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.block_size = block_size
        self.token_embedding_layer = SimpleEmbedding(vocab_size, output_dim)
        self.pos_embedding_layer = SimpleEmbedding(block_size, output_dim)

    def __call__(self, input_ids):
        token_embeddings = self.token_embedding_layer(input_ids)
        print("Token embeddings shape:", token_embeddings.shape)
        pos_embeddings = self.pos_embedding_layer(torch.arange(self.block_size))
        print("Position embeddings shape:", pos_embeddings.shape)
        return token_embeddings + pos_embeddings

    def __repr__(self):
        return f"SimpleGPT2Embedding(vocab_size={self.vocab_size}, output_dim={self.output_dim}, block_size={self.block_size})"


if __name__ == "__main__":
    input_ids = torch.tensor([5, 1, 3, 2])
    vocab_size = 5
    output_dim = 3
    torch.manual_seed(123)
    # embedding = torch.nn.Embedding(vocab_size, output_dim)
    embedding = SimpleEmbedding(vocab_size, output_dim)
    print(embedding.weight)
    print(embedding(torch.tensor([3])))

    from BPETokenizer import GPT2TikTokenizer

    with open("the-verdict.txt", "r", encoding="utf-8") as file:
        text = file.read()
        output_dim = 256
        vocab_size = 50257
        max_length = 4
        block_size = max_length

        tokenizer = GPT2TikTokenizer()
        dataloader = create_data_loader(text, tokenizer, batch_size=8, max_length=max_length, stride=5)
        data_iter = iter(dataloader)
        inputs, targets = next(data_iter)
        print("Input IDs:", inputs)
        print("Input shape:", inputs.shape)

        embedding = SimpleGPT2Embedding(vocab_size, output_dim, block_size)

        input_embeddings = embedding(inputs)
        print("Input embeddings shape:", input_embeddings.shape)

