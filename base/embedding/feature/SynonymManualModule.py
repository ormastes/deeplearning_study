import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from base.embedding.token.FeatureEmbeddingTokenizerMaker import get_symbol_synonyms
from base.embedding.feature.FeatureTokenizer import VirtualTokenizer
from base.embedding.feature.VirtualEmbedding import ReverseEmbedding, VirtualEmbedding
from base.prim.Linear import Linear


import torch

from base.config.Config import FeatureEmbeddingLLM, OTHER_SETTINGS
from base.prim.Embedding import Embedding


class SynonymManualModule(torch.nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.embedding = VirtualEmbedding(tokenizer.vocab_size, config.voca_embed_dim, tokenizer)
        self.to_synonym_embedding = Embedding(tokenizer.vocab_size, config.additional_context_dim)
        self.synonym_embedding = Embedding(config.additional_context_dim, config.voca_embed_dim)
        self.reverse_embedding = ReverseEmbedding(tokenizer.max_id(), config.embed_dim)
        self.synonym_dropout = torch.nn.Dropout(config.drop_rate)

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
        embedding = self.embedding.super().forward(ids)
        embedding = self.embed_in_dropout(embedding)
        synonym_embedding = self.to_synonym_embedding(ids)
        synonym_embedding = self.synonym_embedding(synonym_embedding)
        synonym_embedding = embedding + synonym_embedding
        embedding = torch.cat(
            [synonym_embedding, self.padding[0:len(ids), :]],
            dim=1)
        embedding = self.reverse_embedding(embedding)
        return embedding

# if main
if __name__ == "__main__":
    module_name='SynonymManualModule'
    config = FeatureEmbeddingLLM()

    tokenizer = VirtualTokenizer(config)
    max_id = tokenizer.max_id()

    torch.autograd.set_detect_anomaly(True)
    setting = OTHER_SETTINGS()
    model = SynonymManualModule(config, tokenizer)
    model.to(config.device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay
    )

    # model save exists
    model_file = f"{config.model_path}/{module_name}_model_200.pt"
    if os.path.exists(model_file):
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # copy model save to temp directory with overwrite
        os.system(f"cp {model_file} {config.model_path}/{module_name}_model_200_temp.pt")
        model.to(config.device)

    batch_size = 1024 * 16 * 4
    dataset = []
    for i in range(0, len(get_symbol_synonyms().symbols), batch_size):
        ids = []
        for j in range(i, i + config.context_len):
            if j >= len(get_symbol_synonyms().symbols):
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
            }, f"{config.model_path}/{module_name}_model_{j}.pt")