import os

from base.embedding.token.FeatureEmbeddingTokenizerMaker import get_symbol_synonyms
from base.embedding.feature.FeatureTokenizer import VirtualTokenizer

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


import torch

from base.config.Config import FeatureEmbeddingLLM, OTHER_SETTINGS
from base.prim.Embedding import Embedding



class VirtualEmbedding(Embedding):
    def __init__(self, vocab_size, embedded_dim, tokenizer):
        super().__init__(vocab_size, embedded_dim)
        self.tokenizer = tokenizer
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, _embedding, only_call_super=False):
        _embedding.symbol = super().forward(_embedding.ids)

        if only_call_super:
            return _embedding
        symbol_from_synonyms = torch.zeros_like(_embedding.symbol)
        for j in range(_embedding.ids.shape[0]):
            for i in range(_embedding.length):
                input_id = _embedding.ids[j, i]
                synonym_ids = self.tokenizer.id_to_synonym_id(input_id)
                i_sum = torch.sum(_embedding.symbol[i])
                synonym_embedding_sum = None
                for synonym_id in synonym_ids:
                    synonym_embedding = super().forward(torch.tensor([synonym_id]))
                    d = i_sum/ torch.sum(synonym_embedding)
                    synonym_embedding = self.dropout(d*synonym_embedding)
                    if synonym_embedding_sum is None:
                        synonym_embedding_sum = synonym_embedding
                    else:
                        synonym_embedding_sum = synonym_embedding_sum + synonym_embedding
                if synonym_embedding_sum is not None:
                    symbol_from_synonyms[j, i] = synonym_embedding_sum
        _embedding.symbol_from_synonyms = symbol_from_synonyms
        return _embedding


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
        self.embedding = VirtualEmbedding(tokenizer.max_id(), config.voca_embed_dim, tokenizer)
        self.reverse_embedding = ReverseEmbedding(tokenizer.max_voca_id(), config.voca_embed_dim)
        self.config = config
        self.tokenizer = tokenizer
        self.padding = (torch.tensor([0.1] * config.num_batches * config.additional_context_dim * config.context_len)
                .view(config.num_batches, config.context_len,
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
        from base.embedding.feature.VirtualEmbeddingV5 import EmbeddingValue
        embedding = EmbeddingValue(ids, self.config)
        embedding = self.embedding(embedding)
        _embedding = embedding.symbol_padded
        _embedding = self.reverse_embedding(_embedding)
        return _embedding

    def forward_train(self, ids):
        from base.embedding.feature.VirtualEmbeddingV5 import EmbeddingValue
        embedding = EmbeddingValue(ids, self.config)
        embedding = self.embedding(embedding)
        embedding_mid = torch.cat(
            [embedding.symbol_from_synonyms, self.padding[:ids.shape[0], :ids.shape[1], :]],
            dim=2)
        embedding = self.reverse_embedding(embedding_mid)
        return embedding, embedding_mid

if __name__ == "__main__":
    module_name = 'TestModule'
    max_epoch=1200
    config = FeatureEmbeddingLLM()

    tokenizer = VirtualTokenizer(config)
    max_id = tokenizer.max_id()

    torch.autograd.set_detect_anomaly(True)
    setting = OTHER_SETTINGS()
    model = TestModule(config, tokenizer)

    model.to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay
    )

    # model save exists
    model_file = f"{config.model_path}/{module_name}_model_saved.pt"
    if os.path.exists(model_file):
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # copy model save to temp directory with overwrite
        os.system(f"cp {model_file} {config.model_path}/{module_name}_model_{max_epoch}_temp.pt")
        model.to(config.device)


    batch_size = 1024*16*4
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
    for j in range(max_epoch+1):

        for i, _ in enumerate(dataset):
            # progress
            print(f"epoch: {j}, batch: {i}/{len(dataset)}")
            ids = dataset_tensor #[i]

            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            embedding = model(ids)
            _ids = ids.squeeze(0)
            _embedding = embedding.squeeze(0)
            loss = torch.nn.functional.cross_entropy(_embedding, _ids)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients

        if True: # padding to be zero training
            ids = (torch.tensor([tokenizer.pad_token_id, tokenizer.v_key_id, tokenizer.v_value_id] * config.num_batches * (config.context_len//3)).flatten().
                   view(config.num_batches, (config.context_len//3)*3).to(config.device))

            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            embedding, mid_embedding = model.forward_train(ids)
            mid_expected = torch.full_like(mid_embedding, 0.0)
            _ids = ids.squeeze(0)
            _embedding = embedding.squeeze(0)
            loss = (torch.nn.functional.cross_entropy(_embedding, _ids)
                    + torch.nn.functional.mse_loss(mid_embedding, mid_expected))
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients


        if j % 10 == 0:
            model.eval()
            loss_sum = 0
            accracy_sum = 0
            times = 0
            with torch.no_grad():
                for i, _ in enumerate(dataset):
                    ids = dataset_tensor #[i]
                    embedding = model(ids)
                    _ids = ids.squeeze(0)
                    _embedding = embedding.squeeze(0)
                    loss = torch.nn.functional.cross_entropy(_embedding, _ids)
                    loss_sum += loss
                    accuracy = torch.sum(torch.argmax(embedding, dim=2) == ids).float() / len(
                        ids.flatten())
                    accracy_sum += accuracy
                    times += 1
            print(f"loss: {loss_sum / times}, accuracy: {accracy_sum / times}")
            model.train()
            # save model with optimizer
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f"{config.model_path}/{module_name}_model_{j}.pt")