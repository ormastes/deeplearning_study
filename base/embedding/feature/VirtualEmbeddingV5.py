import os
from typing import Mapping, Any

from torch.nn.modules.module import T

from base.embedding.token.FeatureEmbeddingTokenizerMaker import get_symbol_synonyms
from base.embedding.feature.FeatureTokenizer import VirtualTokenizer
from base.prim.Linear import Linear

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import torch

from base.config.Config import FeatureEmbeddingLLM, OTHER_SETTINGS
from base.prim.Embedding import Embedding
from base.embedding.feature.VirtualEmbedding import VirtualEmbedding, ReverseEmbedding


class _VirtualEmbeddingV5(Embedding):
    def __init__(self, id_voca_size, id_synonym_size, voca_embed_dim, syno_embed_dim, tokenizer):
        super().__init__(id_voca_size, voca_embed_dim)
        self.id_voca_size = id_voca_size
        self.synonym_voca_size = id_synonym_size
        self.tokenizer = tokenizer
        self.dropout = torch.nn.Dropout(0.3)
        self.synonym_id_embedding_out = Linear(syno_embed_dim, voca_embed_dim)
        self.synonym_id_sum_out = Linear(syno_embed_dim, id_synonym_size)
        self.scale = 8.0

    def three_stage(self, x):
        scale = self.scale
        x = x * scale
        steepness = 3.0
        positive_x = x - 4.0
        negative_x = -x - 4.0
        negative = torch.nn.functional.sigmoid(steepness * negative_x)
        positive = torch.nn.functional.sigmoid(steepness * positive_x)
        return positive - negative

    # load store model
    def load_state_dict(self, state_dict, strict=True):
        self.weight = state_dict['weight']
        self.synonym_id_embedding_out.load_state_dict(state_dict['synonym_id_embedding_out'])
        self.synonym_id_sum_out.load_state_dict(state_dict['synonym_id_sum_out'])
        self.synonym_id_embedding_out.requires_grad_(False)
        self.synonym_id_sum_out.requires_grad_(False)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'weight': self.weight,
            'synonym_id_embedding_out': self.synonym_id_embedding_out.state_dict(),
            'synonym_id_sum_out': self.synonym_id_sum_out.state_dict()
        }
        return state_dict


    def forward(self, _embedding, no_synonym_id_sum=False):
        #embedding = self.v1.super().forward(input_ids)
        _embedding.synonyms = super().forward(_embedding.ids) / self.scale

        embedding, _synonym_id_sum = self.forward_synonym_embedding(_embedding)

        if no_synonym_id_sum:
            expected_synonym_id_sum = None
        else:
            expected_synonym_id_sum = torch.zeros_like(_synonym_id_sum)
            for j in range(_embedding.ids.shape[0]):
                for i in range(_embedding.ids.shape[1]):
                    input_id = _embedding.ids[j, i]
                    synonym_ids = self.tokenizer.id_to_synonym_id(input_id)
                    for synonym_id in synonym_ids:
                        one_hot = torch.nn.functional.one_hot(torch.tensor(synonym_id - self.id_voca_size),
                                                              num_classes=self.synonym_voca_size).to(self.weight.dtype)
                        if one_hot.device != self.weight.device:
                            one_hot = one_hot.to(self.weight.device)
                        expected_synonym_id_sum[j, i] = expected_synonym_id_sum[j, i] + one_hot


        return _embedding, _synonym_id_sum, expected_synonym_id_sum

    def forward_synonym_embedding(self, _embedding):
        _embedding.synonyms = self.three_stage(_embedding.synonyms)
        # embedding = embedding + self.synonym_id_embedding_out(synonym_id_embedding)
        _embedding.symbol_from_synonyms  = self.synonym_id_embedding_out(_embedding.synonyms)
        _synonym_id_sum = self.synonym_id_sum_out(_embedding.synonyms)

        return _embedding, _synonym_id_sum


class EmbeddingValue():
    def __init__(self, ids, config):
        self.config = config
        self.ids = ids
        self.length = len(ids)
        self.symbol = None
        self.symbol_padded = None
        self.symbol_id_one_hot = None
        self.synonyms = None
        self.language = None
        self._symbol_from_synonyms = None


    @property
    def embedding(self):
        return torch.cat([self.symbol_padded, self.language,  self.synonyms], dim=2)

    @embedding.setter
    def embedding(self, value):
        self.symbol_padded = value[:, :, 0:self.config.voca_embed_dim]
        self.language = value[:, :, self.config.voca_embed_dim:self.config.voca_embed_dim + self.config.language.TOTAL_SIZE]
        self.synonyms = value[:, :, self.config.voca_embed_dim + self.config.language.TOTAL_SIZE:]
    # set self.symbol_from_synonyms
    @property
    def symbol_from_synonyms(self):
        return self._symbol_from_synonyms

    @symbol_from_synonyms.setter
    def symbol_from_synonyms(self, value):
        self._symbol_from_synonyms = value
        self.symbol = self.symbol + value
        pad = (torch.tensor([0.1] * self.symbol.shape[0] * self.symbol.shape[1] * self.config.additional_context_dim)
                                      .view(self.symbol.shape[0], self.symbol.shape[1], self.config.additional_context_dim)
                                      .to(self.config.device))
        self.symbol_padded = torch.cat([self.symbol, pad], dim=2)
        self.language = (torch.tensor([0.1] * self.symbol_padded.shape[0] * self.symbol_padded.shape[1]
                                      * self.config.language.TOTAL_SIZE).
                         view(self.symbol_padded.shape[0], self.symbol_padded.shape[1], self.config.language.TOTAL_SIZE).
                         to(self.config.device))


class VirtualEmbeddingV5(torch.nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.embedding_v1 = VirtualEmbedding(tokenizer.max_id(), config.voca_embed_dim, tokenizer)
        self.embedding_v1.requires_grad_(False)
        self.embedding_v2 = _VirtualEmbeddingV5(tokenizer.max_voca_id(), tokenizer.max_synonym_id(),
                                                config.voca_embed_dim, config.syno_embed_dim, tokenizer)
        self.reverse_embedding = ReverseEmbedding(tokenizer.max_voca_id(), config.voca_embed_dim)
        self.reverse_embedding.requires_grad_(False)
        self.config = config
        self.tokenizer = tokenizer
        self.padding = (torch.tensor([0.1] * config.num_batches * config.additional_context_dim * config.context_len)
        .view(config.num_batches, config.context_len,
              config.additional_context_dim).to(
            config.device)).requires_grad_(False)

    # load store
    def load_state_dict(self, state_dict, strict=True):
        self.embedding_v1.load_state_dict(state_dict['embedding'])
        if 'embedding_v1' in state_dict:
            self.embedding_v2.load_state_dict(state_dict['embedding_v2'])
        self.reverse_embedding.load_state_dict(state_dict['reverse_embedding'])
        self.embedding_v1.requires_grad_(False)
        self.reverse_embedding.requires_grad_(False)

    def train(self: T, mode: bool = True) -> T:
        self.embedding_v1.train(False)
        self.embedding_v2.train(mode)
        self.reverse_embedding.train(False)
        return self

    def forward_synonym_embedding(self, prev_synonym_embedding, embedding):
        #prev_synonym_embedding, _, _, _= self.embedding_v2.forward_synonym_embedding(prev_synonym_id_embedding)
        embedding, _ = self.embedding_v2.forward_synonym_embedding(embedding)
        return embedding.symbol_from_synonyms - prev_synonym_embedding, embedding

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'embedding': self.embedding_v1.state_dict(),
            'embedding_v2': self.embedding_v2.state_dict(),
            'reverse_embedding': self.reverse_embedding.state_dict()
        }
        return state_dict

    def forward(self, ids, no_synonym_id_sum=False, with_reverse_embedding=True):
        embedding = EmbeddingValue(ids, self.config)
        embedding = self.embedding_v1.forward(embedding, only_call_super=True)
        embedding, synonym_id_sum, expected_synonym_id_sum = self.embedding_v2(embedding, no_synonym_id_sum)
        if with_reverse_embedding:
            embedding.symbol_id_one_hot = self.reverse_embedding(embedding.symbol_padded)
        return embedding, synonym_id_sum, expected_synonym_id_sum


if __name__ == "__main__":
    from grokfast import gradfilter_ma, gradfilter_ema

    module_name = 'TestModuleV5'
    max_epoch = 2400
    config = FeatureEmbeddingLLM()

    tokenizer = VirtualTokenizer(config)
    max_id = tokenizer.max_id()

    torch.autograd.set_detect_anomaly(True)
    setting = OTHER_SETTINGS()
    model = VirtualEmbeddingV5(config, tokenizer)

    model.to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay
    )
    #model2 = VirtualEmbeddingV5(config, tokenizer)
    #model2.to(config.device)
    # model save exists
    #model_file = f"{config.model_path}/{module_name}_model_saved.pt"
    model_file = f"{config.model_path}/{module_name}_model_2400.pt"
    #model_file = f"{config.model_path}/TestModule_model_saved.pt"
    if os.path.exists(model_file):
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        #checkpoint = torch.load(model_file1)
        #model.load_state_dict(checkpoint['model_state_dict'])
        print(f"load model from {model_file}")
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # copy model save to temp directory with overwrite
        os.system(f"cp {model_file} {config.model_path}/{module_name}_model_{max_epoch}_temp.pt")

        model.to(config.device)
        # compare weights of model and model2
        if False:#for i, (p1, p2) in enumerate(zip(model.parameters(), model2.parameters())):

            if torch.all(torch.eq(p1, p2)):
                print(f"param {i} is same")
            else:
                print(f"param {i} is different")

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
    mse_loss = torch.nn.MSELoss()
    model.train()  # Set model to training mode
    grads = None
    for j in range(max_epoch + 1):

        for i, _ in enumerate(dataset):
            # progress
            print(f"epoch: {j}, batch: {i}/{len(dataset)}")
            ids = dataset_tensor #[i]

            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            embedding, synonym_id_sum, expected_synonym_id_sum= model(ids)
            will_be_prev_synonym_embedding = embedding.symbol_from_synonyms
            _ids = ids.squeeze(0)
            _symbol_id_one_hot = embedding.symbol_id_one_hot.squeeze(0)
            loss = torch.nn.functional.cross_entropy(_symbol_id_one_hot, _ids)
            loss = loss + mse_loss(synonym_id_sum, expected_synonym_id_sum)
            loss.backward()  # Calculate loss gradients
            grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
            optimizer.step()  # Update model weights using loss gradients

        if j % 10 == 0:
            model.eval()
            loss_sum = 0
            accracy_sum = 0
            times = 0
            with torch.no_grad():
                for i, _ in enumerate(dataset):
                    ids = dataset_tensor
                    embedding, synonym_id_sum, expected_synonym_id_sum = model(ids)
                    will_be_prev_synonym_embedding = embedding.symbol_from_synonyms
                    _ids = ids.squeeze(0)
                    _symbol_id_one_hot = embedding.symbol_id_one_hot.squeeze(0)
                    loss = torch.nn.functional.cross_entropy(_symbol_id_one_hot, _ids)
                    loss = loss + mse_loss(synonym_id_sum, expected_synonym_id_sum)
                    loss_sum += loss
                    accuracy = torch.sum(torch.argmax(embedding.symbol_id_one_hot, dim=2) == ids).float() / len(ids.flatten())
                    accracy_sum += accuracy
                    times += 1
            print(f"loss: {loss_sum / times}, accuracy: {accracy_sum / times}")
            model.train()
            # save model with optimizer
            save_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(save_state, f"{config.model_path}/{module_name}_model_{j}.pt")
            #load_state = torch.load(f"{config.model_path}/{module_name}_model_{j}.pt")
            #model2.load_state_dict(load_state['model_state_dict'])
            if False: #for i, (p1, p2) in enumerate(zip(model.parameters(), model2.parameters())):

                if torch.all(torch.eq(p1, p2)):
                    print(f"param {i} is same")
                else:
                    print(f"param {i} is different")
            #model.load_state_dict(load_state['model_state_dict'])
            if False:#for i, (p1, p2) in enumerate(zip(model.parameters(), model2.parameters())):

                if torch.all(torch.eq(p1, p2)):
                    print(f"param {i} is same")
                else:
                    print(f"param {i} is different")
