import pandas as pd
import torch
from tokenizers import Tokenizer

from base.config.Config import GPT2_CONFIG_124M, OTHER_SETTINGS
from base.prim.Embedding import Embedding


class FeatureEmbeddingLLM(GPT2_CONFIG_124M):
    def __init__(self):
        super().__init__()
        self.feature_embedding_dir = f"{self.data_path}/feature_embedding"
        self.synonyms = pd.read_csv(f"{self.data_path}/feature_embedding/synonyms.csv")
        self.vocab_size = 90000
        self.additional_context_dim = 10


class Synonym:
    def __init__(self, part_of_speech, merged_synonyms):
        self.part_of_speech = part_of_speech
        self.merged_synonyms = merged_synonyms


class SymbolSynonym:
    def __init__(self, symbols, synonyms, synonymed_symbols):
        self.symbols = symbols
        self.synonyms = synonyms
        self.synonymed_symbols = synonymed_symbols


config = FeatureEmbeddingLLM()

import pickle

symbols_synonyms_file = f"{config.feature_embedding_dir}/symbols_synonyms.pkl"

# load the total_synonyms and total_symbols
with open(symbols_synonyms_file, "rb") as f:
    symbol_synonyms = pickle.load(f)

tokenizer = Tokenizer.from_file("MyBPETokenizer.json")


class VirtualTokenizer:
    def __init__(self, tokenizer, symbols, synonyms, synonymed_symbols, config):
        self.tokenizer = tokenizer
        self.symbols = symbols
        self.synonyms = synonyms
        self.synonymed_symbols = synonymed_symbols
        self.vocab_size = config.vocab_size
        self.symbol_id_to_synonym_id = [[] for _ in range(len(symbols))]
        for si, synonym in enumerate(self.synonyms):
            for symbol in synonym.merged_synonyms:
                id = self.token_to_id(symbol)
                self.symbol_id_to_synonym_id[id].append(self.vocab_size + si)

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def max_id(self):
        return self.vocab_size + len(self.synonyms)

    def id_to_synonym_id(self, id):
        if id >= len(self.symbol_id_to_synonym_id):
            return []
        return self.symbol_id_to_synonym_id[id]

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)


effective_embed_dim = config.embed_dim - config.additional_context_dim
config.effect_embed_dim = effective_embed_dim

tokenizer = VirtualTokenizer(tokenizer, symbol_synonyms.symbols, symbol_synonyms.synonyms,
                             symbol_synonyms.synonymed_symbols, config)
max_id = tokenizer.max_id()


class VirtualEmbedding(Embedding):
    def __init__(self, vocab_size, embedded_dim, tokenizer):
        super().__init__(vocab_size, embedded_dim)
        self.tokenizer = tokenizer

    def forward(self, input_ids):
        embedding = super().forward(input_ids)
        for i in range(len(input_ids)):
            input_id = input_ids[i]
            synonym_ids = self.tokenizer.id_to_synonym_id(input_id)
            for synonym_id in synonym_ids:
                synonym_embedding = super().forward(torch.tensor([synonym_id]))
                embedding[i] = torch.max(embedding[i], synonym_embedding)
        return embedding


class ReverseEmbedding(Embedding):
    def __init__(self, vocab_size, embedded_dim):
        super().__init__(embedded_dim, vocab_size)
        self.revserse_vocab_size = vocab_size

    def forward(self, input):
        recovered = input @ self.weight
        return recovered


class TestModule(torch.nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.embedding = VirtualEmbedding(config.vocab_size, config.embed_dim, tokenizer)
        self.reverse_embedding = ReverseEmbedding(config.vocab_size, config.embed_dim)
        self.config = config
        self.tokenizer = tokenizer
        self.padding = (torch.tensor([0.1] * config.additional_context_dim * config.context_len)
                .view(config.additional_context_dim,
                      config.context_len).to(
                    config.device))

    def forward(self, ids):
        synonym_embedding = self.embedding(ids)
        embedding = torch.cat(
            (synonym_embedding, self.padding),
            dim=1)
        embedding = self.reverse_embedding(embedding)
        return embedding


setting = OTHER_SETTINGS()
model = TestModule(config, tokenizer)
model.to(config.device)  # no assignment model = model.to(device) necessary for nn.Module classes
optimizer = torch.optim.AdamW(
    model.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay
)
batch_size = 128
dataset = []
for i in range(0, len(symbol_synonyms.symbols), batch_size):
    ids = []
    for j in range(i, i + batch_size):
        if j >= len(symbol_synonyms.symbols):
            ids.append(0)
        else:
            ids.append(j)

    dataset.append(ids)
dataset_tensor = torch.tensor(dataset).to(config.device)

model.train()  # Set model to training mode
for j in range(100):

    for i, _ in enumerate(dataset):
        ids = dataset_tensor[i]

        optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
        embedding = model(ids)
        loss = torch.nn.functional.cross_entropy(embedding, ids)
        loss.backward()  # Calculate loss gradients
        optimizer.step()  # Update model weights using loss gradients

    if j % 10 == 0:
        model.eval()
        loss_sum = 0
        accracy_sum = 0
        times = 0
        with torch.no_grad():
            for i, _ in enumerate(dataset):
                ids = dataset_tensor[i]
                embedding = model(ids)
                loss = torch.nn.functional.cross_entropy(embedding, ids)
                loss_sum += loss
                accracy = torch.sum(torch.argmax(embedding, dim=1) == ids).float() / len(ids)
                accracy_sum += accracy
                times += 1
        print(f"loss: {loss_sum / times}, accracy: {accracy_sum / times}")
        model.train()
