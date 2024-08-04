import os

from base.prim.Linear import Linear

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

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

config.core_embed_dim = config.embed_dim
effective_embed_dim = config.core_embed_dim - config.additional_context_dim
config.effect_embed_dim = effective_embed_dim

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


tokenizer = VirtualTokenizer(tokenizer, symbol_synonyms.symbols, symbol_synonyms.synonyms,
                             symbol_synonyms.synonymed_symbols, config)
max_id = tokenizer.max_id()


class VirtualEmbedding(Embedding):
    def __init__(self, vocab_size, embedded_dim, tokenizer):
        super().__init__(vocab_size, embedded_dim)
        self.tokenizer = tokenizer
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, input_ids):
        embedding = super().forward(input_ids)
        for i in range(len(input_ids)):
            input_id = input_ids[i]
            synonym_ids = self.tokenizer.id_to_synonym_id(input_id)
            i_sum = torch.sum(embedding[i])
            for synonym_id in synonym_ids:
                synonym_embedding = super().forward(torch.tensor([synonym_id]))
                d = i_sum/ torch.sum(synonym_embedding)
                embedding[i] = embedding[i] + self.dropout(d*synonym_embedding)
            # embedding[i] = embedding[i] / (len(synonym_ids) + 1)
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
        self.embedding = VirtualEmbedding(tokenizer.max_id(), config.effect_embed_dim, tokenizer)
        self.reverse_embedding = ReverseEmbedding(tokenizer.max_id(), config.embed_dim)
        self.config = config
        self.tokenizer = tokenizer
        self.padding = (torch.tensor([0.1] * config.additional_context_dim * config.context_len)
                .view(config.context_len,
                      config.additional_context_dim).to(
                    config.device))

    # load store
    def load_state_dict(self, state_dict, strict=True):
        self.embedding.load_state_dict(state_dict['embedding'])
        self.reverse_embedding.load_state_dict(state_dict['reverse_embedding'])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'embedding': self.embedding.state_dict(),
            'reverse_embedding': self.reverse_embedding.state_dict()
        }
        return state_dict

    def forward(self, ids):
        synonym_embedding = self.embedding(ids)
        embedding = torch.cat(
            [synonym_embedding, self.padding[0:len(ids), :]],
            dim=1)
        embedding = self.reverse_embedding(embedding)
        return embedding


class SynonymModule(torch.nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.embedding = Embedding(tokenizer.vocab_size, config.effect_embed_dim)
        self.embed_in_dropout = torch.nn.Dropout(config.drop_rate)
        self.to_synonym_embedding = Embedding(tokenizer.vocab_size, config.additional_context_dim)
        self.synonym_dropout = torch.nn.Dropout(config.drop_rate)
        self.embed_out_dropout = torch.nn.Dropout(config.drop_rate)
        self.synonym_id_recovers = Linear(config.additional_context_dim, tokenizer.max_id()-config.embed_dim)
        self.synonym_embedding = Linear(config.additional_context_dim, config.effect_embed_dim)
        self.reverse_embedding = ReverseEmbedding(tokenizer.max_id(), config.embed_dim)
        self.config = config
        self.tokenizer = tokenizer
        self.padding = (torch.tensor([0.1] * config.additional_context_dim * config.context_len)
                .view(config.context_len,
                      config.additional_context_dim).to(
                    config.device))

    def train(self, mode=True):
        self.to_synonym_embedding.train(mode)
        self.synonym_embedding.train(mode)

    def eval(self):
        self.to_synonym_embedding.eval()
        self.synonym_embedding.eval()

    # load store
    def load_state_dict(self, state_dict, strict=True):
        embedding = state_dict['embedding']
        # make embedding weight size to be the same as the current embedding
        embedding['weight'] = embedding['weight'][0:self.embedding.weight.shape[0], :]
        self.embedding.load_state_dict(embedding)
        self.reverse_embedding.load_state_dict(state_dict['reverse_embedding'])
        if 'to_synonym_embedding' in state_dict:
            self.to_synonym_embedding.load_state_dict(state_dict['to_synonym_embedding'])
            self.synonym_embedding.load_state_dict(state_dict['synonym_embedding'])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'embedding': self.embedding.state_dict(),
            'reverse_embedding': self.reverse_embedding.state_dict(),
            'to_synonym_embedding': self.to_synonym_embedding.state_dict(),
            'synonym_embedding': self.synonym_embedding.state_dict()
        }
        return state_dict

    def forward(self, ids):
        embedding = self.embedding.forward(ids)
        embedding = self.embed_in_dropout(embedding)
        synonym_embedding = self.to_synonym_embedding(ids)
        synonym_embedding = self.synonym_dropout(synonym_embedding)
        if self.training:
            synonym_hot = self.synonym_id_recovers(synonym_embedding).to(self.config.device)
            expected_synonym_hots = None
            for i in range(len(ids)):
                input_id = ids[i]
                synonym_ids = self.tokenizer.id_to_synonym_id(input_id)
                synonym_ids = [id - self.config.embed_dim for id in synonym_ids]
                synonym_ids = torch.tensor(synonym_ids).to(self.config.device)
                expected_synonym_hot = torch.zeros_like(synonym_hot[0])
                expected_synonym_hot.scatter_(0, synonym_ids, 1.0)
                expected_synonym_hots = torch.cat([expected_synonym_hots, expected_synonym_hot.unsqueeze(0)]) if expected_synonym_hots is not None else expected_synonym_hot.unsqueeze(0)

        synonym_embedding = self.synonym_embedding(synonym_embedding)
        synonym_embedding = embedding + synonym_embedding

        embedding = torch.cat(
            [synonym_embedding, self.padding[0:len(ids), :]],
            dim=1)
        embedding = self.embed_out_dropout(embedding)
        embedding = self.reverse_embedding(embedding)

        return embedding

torch.autograd.set_detect_anomaly(True)
setting = OTHER_SETTINGS()
model = SynonymModule(config, tokenizer)
model.to(config.device)  # no assignment model = model.to(device) necessary for nn.Module classes
optimizer = torch.optim.AdamW(
    model.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay
)

# model save exists
if os.path.exists(f"{config.model_path}/embed_2_model_saved.pt"):
    checkpoint = torch.load(f"{config.model_path}/embed_2_model_saved.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # copy model save to temp directory with overwrite
    os.system(f"cp {config.model_path}/embed_2_model_200.pt {config.model_path}/embed_2_model_200_temp.pt")
    model.to(config.device)


batch_size = 1024*16*4
dataset = []
for i in range(0, len(symbol_synonyms.symbols), batch_size):
    ids = []
    for j in range(i, i + config.context_len):
        if j >= len(symbol_synonyms.symbols):
            ids.append(0)
        else:
            ids.append(j)

    dataset.append(ids)
dataset_tensor = torch.tensor(dataset).to(config.device)

model.train()  # Set model to training mode
for j in range(201):

    for i, _ in enumerate(dataset):
        # progress
        print(f"epoch: {j}, batch: {i}/{len(dataset)}")
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
        print(f"loss: {loss_sum / times}, accuracy: {accracy_sum / times}")
        model.train()
        # save model with optimizer
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f"{config.model_path}/embed_2_model_{j}.pt")


