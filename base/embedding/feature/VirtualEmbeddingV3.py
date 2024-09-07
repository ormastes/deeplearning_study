import os

from base.embedding.token.FeatureEmbeddingTokenizerMaker import get_symbol_synonyms
from base.embedding.feature.FeatureTokenizer import VirtualTokenizer
from base.prim.Linear import Linear

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


import torch

from base.config.Config import FeatureEmbeddingLLM, OTHER_SETTINGS
from base.prim.Embedding import Embedding
from base.embedding.feature.VirtualEmbedding import VirtualEmbedding, ReverseEmbedding


class VirtualEmbeddingV3(Embedding):
    def __init__(self, id_voca_size, synonym_voca_size, embedded_dim, tokenizer):
        super().__init__(id_voca_size, embedded_dim)
        self.id_voca_size = id_voca_size
        self.synonym_voca_size = synonym_voca_size
        self.tokenizer = tokenizer
        self.dropout = torch.nn.Dropout(0.3)
        self.synonym_id_embedding_out = Linear(embedded_dim, embedded_dim)
        self.synonym_id_sum_out = Linear(embedded_dim, synonym_voca_size)

    def three_stage(self, x):
        positive_x = x - 4.0
        negative_x = -x - 4.0
        negative = torch.nn.functional.sigmoid(negative_x)
        positive = torch.nn.functional.sigmoid(positive_x)
        return positive - negative
    def forward(self, input_ids):
        #embedding = self.v1.super().forward(input_ids)
        synonym_id_embedding = super().forward(input_ids)
        synonym_id_embedding = self.three_stage(synonym_id_embedding)
        #embedding = embedding + self.synonym_id_embedding_out(synonym_id_embedding)
        embedding = self.synonym_id_embedding_out(synonym_id_embedding)
        expected_synonym_id_sum = self.synonym_id_sum_out(synonym_id_embedding)
        synonym_id_sum = (torch.tensor([0.0] * len(input_ids) * self.synonym_voca_size).to(self.weight.dtype)
                          .to(self.weight.device)).view(len(input_ids), self.synonym_voca_size)

        for i in range(len(input_ids)):
            input_id = input_ids[i]
            synonym_ids = self.tokenizer.id_to_synonym_id(input_id)
            for synonym_id in synonym_ids:
                one_hot = torch.nn.functional.one_hot(torch.tensor([synonym_id- self.id_voca_size]), num_classes=self.synonym_voca_size).to(self.weight.dtype)
                if one_hot.device != self.weight.device:
                    one_hot = one_hot.to(self.weight.device)
                synonym_id_sum[i] = synonym_id_sum[i] + one_hot

        return embedding, synonym_id_embedding, synonym_id_sum, expected_synonym_id_sum


class TestModuleV3(torch.nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.embedding_v1 = VirtualEmbedding(tokenizer.max_id(), config.voca_embed_dim, tokenizer)
        self.embedding_v1.requires_grad_(False)
        self.embedding_v2 = VirtualEmbeddingV3(tokenizer.max_voca_id(), tokenizer.max_synonym_id(), config.voca_embed_dim, tokenizer)
        self.reverse_embedding = ReverseEmbedding(tokenizer.max_id(), config.embed_dim)
        self.reverse_embedding.requires_grad_(False)
        self.config = config
        self.tokenizer = tokenizer
        self.padding = (torch.tensor([0.1] * config.additional_context_dim * config.context_len)
                .view(config.context_len,
                      config.additional_context_dim).to(
                    config.device))
    # load store
    def load_state_dict(self, state_dict, strict=True):
        self.embedding_v1.load_state_dict(state_dict['embedding'])
        if 'embedding_v1' in state_dict:
            self.embedding_v2.load_state_dict(state_dict['embedding_v2'])
        self.reverse_embedding.load_state_dict(state_dict['reverse_embedding'])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'embedding': self.embedding_v1.state_dict(),
            'embedding_v2': self.embedding_v2.state_dict(),
            'reverse_embedding': self.reverse_embedding.state_dict()
        }
        return state_dict

    def forward(self, ids):
        synonym_embedding = self.embedding_v1.forward(ids, only_call_super=True)
        synonym_embedding_v2, synonym_id_embedding, synonym_id_sum, expected_synonym_id_sum = self.embedding_v2(ids)
        synonym_embedding = synonym_embedding + synonym_embedding_v2
        embedding = torch.cat(
            [synonym_embedding, self.padding[0:len(ids), :]],
            dim=1)
        embedding = self.reverse_embedding(embedding)
        return embedding, synonym_id_embedding, synonym_id_sum, expected_synonym_id_sum

if __name__ == "__main__":
    module_name = 'TestModuleV3'
    max_epoch=1200
    config = FeatureEmbeddingLLM()

    tokenizer = VirtualTokenizer(config)
    max_id = tokenizer.max_id()

    torch.autograd.set_detect_anomaly(True)
    setting = OTHER_SETTINGS()
    model = TestModuleV3(config, tokenizer)

    model.to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay
    )

    # model save exists
    #model_file = f"{config.model_path}/{module_name}_model_saved.pt"
    model_file = f"{config.model_path}/TestModuleV2_model_saved.pt"
    if os.path.exists(model_file):
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
    mse_loss = torch.nn.MSELoss()
    model.train()  # Set model to training mode
    for j in range(max_epoch+1):

        for i, _ in enumerate(dataset):
            # progress
            print(f"epoch: {j}, batch: {i}/{len(dataset)}")
            ids = dataset_tensor[i]

            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            embedding, synonym_id_embedding, synonym_id_sum, expected_synonym_id_sum = model(ids)
            loss = torch.nn.functional.cross_entropy(embedding, ids) + mse_loss(synonym_id_sum, expected_synonym_id_sum)
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
                    embedding, synonym_id_embedding, synonym_id_sum, expected_synonym_id_sum = model(ids)
                    loss = torch.nn.functional.cross_entropy(embedding, ids) + mse_loss(synonym_id_sum, expected_synonym_id_sum)
                    loss_sum += loss
                    accuracy = torch.sum(torch.argmax(embedding, dim=1) == ids).float() / len(ids)
                    accracy_sum += accuracy
                    times += 1
            print(f"loss: {loss_sum / times}, accuracy: {accracy_sum / times}")
            model.train()
            # save model with optimizer
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f"{config.model_path}/{module_name}_model_{j}.pt")