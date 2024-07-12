import torch

from base.dataset.SimpleDataset import create_data_loader
from base.gpt.BPETokenizer import GPT2TikTokenizer
from base.prim.Embedding import Embedding
from base.prim.SequencePostionEmbedding import SequencePositionEmbedding
from base.util.Log import Logger


class SimpleGPT2Embedding(torch.nn.Module):
    def __init__(self, vocab_size, embedded_dim, context_length):
        super(SimpleGPT2Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedded_dim = embedded_dim
        self.context_length = context_length
        self.token_embed = Embedding(vocab_size, embedded_dim)
        self.pos_embed = SequencePositionEmbedding(context_length, embedded_dim)
        self.log = Logger.get_instance()

    def forward(self, input_ids):
        token_embeddings = self.token_embed(input_ids)
        self.log.debug("Token embeddings shape:", token_embeddings.shape)
        pos_embeddings = self.pos_embed(token_embeddings)
        self.log.debug("Position embeddings shape:", pos_embeddings.shape)
        return token_embeddings + pos_embeddings

    def __repr__(self):
        return (f"SimpleGPT2Embedding(vocab_size={self.vocab_size}, embedded_dim={self.embedded_dim}, "
                f"context_length={self.context_length})")


if __name__ == "__main__":
    input_ids = torch.tensor([5, 1, 3, 2])
    vocab_size = 5
    output_dim = 3
    torch.manual_seed(123)
    # embedding = torch.nn.Embedding(vocab_size, output_dim)
    embedding = Embedding(vocab_size, output_dim)
    print(embedding.weight)
    print(embedding(torch.tensor([3])))

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