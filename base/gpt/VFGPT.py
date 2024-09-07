import os

import torch.nn.functional
from datasets import load_dataset

from base.config.Config import FeatureEmbeddingLLM, OTHER_SETTINGS
from base.embedding.AttentionLinearBiasPositionalEmbedding import AttentionLinearBiasPositionalEmbedding
from base.embedding.feature.FeatureTokenizer import VirtualTokenizer
from base.embedding.feature.VirtualEmbeddingV5 import VirtualEmbeddingV5
from base.gpt.FeatureAttention import FeatureAttention
from base.prim.FeedForward import FeedForward

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from torch import nn
# Convert the dataset into a PyTorch DataLoader
from torch.utils.data import DataLoader
import pickle
from base.gpt.MultiHeadAttention import MultiHeadAttention
from base.gpt.SimpleGPT2Embedding import SimpleGPT2Embedding
from base.gpt.BPETokenizer import GPT2TikTokenizer
from base.config.GPTConfig import GPT2_CONFIG_124M
from base.util.Util import *
from base.util.Log import *
from base.prim.LayerNorm import LayerNorm
from base.gpt.TransformerBlock import TransformerBlock
from base.prim.Activator import GELU
from base.prim.Linear import Linear


class VFTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = Logger.get_instance()
        self.config = config

        self.norm1 = LayerNorm(config.embed_dim)

        self.attnN1 = config.attention(config=config)
        self.attnN2 = config.attention(config=config)
        self.attnV = config.attention(config=config)

        self.subject_after = nn.Sequential(Linear(config.embed_dim, config.embed_dim_ff_dim),GELU())
        self.verb_after = nn.Sequential(Linear(config.embed_dim, config.embed_dim_ff_dim), GELU())
        self.object_after = nn.Sequential(Linear(config.embed_dim, config.embed_dim_ff_dim), GELU())

        self.embed_after = nn.Sequential(Linear(config.embed_dim_ff_dim*3, config.embed_dim+1), GELU())
        self.norm2 = LayerNorm(config.embed_dim)

        # feed forward, mlp = multi-layer perceptron
        self.mlp = FeedForward(config)

        self.drop = nn.Dropout(config.drop_rate)
        self.front_norm = True

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            prefix+'norm1': self.norm1.state_dict(),
            prefix+'attnN1': self.attnN1.state_dict(),
            prefix + 'attnN2': self.attnN1.state_dict(),
            prefix + 'attnV': self.attnN1.state_dict(),
            prefix + 'subject_after': self.subject_after.state_dict(),
            prefix + 'verb_after': self.verb_after.state_dict(),
            prefix + 'object_after': self.object_after.state_dict(),
            prefix + 'embed_after': self.embed_after.state_dict(),
            prefix+'norm2': self.norm2.state_dict(),
            prefix+'mlp': self.mlp.state_dict(),
            prefix+'drop': self.drop.state_dict()
        }
        if self.config.is_feature_attention:
            state_dict['feature_attnN1'] = self.feature_attnN1.state_dict()
            state_dict['feature_attnN2'] = self.feature_attnN1.state_dict()
            state_dict['feature_attnV'] = self.feature_attnN1.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.norm1.load_state_dict(state_dict['norm1'])
        self.attnN1.load_state_dict(state_dict['attnN1'])
        self.attnN2.load_state_dict(state_dict['attnN2'])
        self.attnV.load_state_dict(state_dict['attnV'])
        self.subject_after.load_state_dict(state_dict['subject_after'])
        self.verb_after.load_state_dict(state_dict['verb_after'])
        self.object_after.load_state_dict(state_dict['object_after'])
        self.embed_after.load_state_dict(state_dict['embed_after'])
        self.norm2.load_state_dict(state_dict['norm2'])
        self.mlp.load_state_dict(state_dict['mlp'])
        self.drop.load_state_dict(state_dict['drop'])
        if self.config.is_feature_attention:
            self.feature_attnN1.load_state_dict(state_dict['feature_attnN1'])
            self.feature_attnN2.load_state_dict(state_dict['feature_attnN2'])
            self.feature_attnV.load_state_dict(state_dict['feature_attnV'])


    def forward(self, x, local_attention_scores=None):
        self.log.shape("Block input", x, x.shape)
        shortcut = x
        if self.front_norm :
            x = self.norm1(x)
        if self.config.attention_window > 0:
            n1 = self.attnN1(x, local_attention_scores)
            n2 = self.attnN2(n1, local_attention_scores)
            v = self.attnV(n2, local_attention_scores)
        else:
            n1 = self.attnN1(x)
            n2 = self.attnN2(n1)
            v = self.attnV(n2)
        svo = torch.cat([self.subject_after(n1), self.verb_after(v), self.object_after(n2)], dim=2)
        ovs = torch.cat([self.object_after(n2), self.verb_after(v), self.subject_after(n1)], dim=2)
        svo = self.embed_after(svo)
        svo = svo[:, :, :-1]
        svo_selector = svo[:, :, -1:]
        ovs = self.embed_after(ovs)
        ovs = ovs[:, :, :-1]
        ovs_selector = ovs[:, :, -1:]
        # select svo or ovs by selector like softmax
        x = x + svo_selector * svo + ovs_selector * ovs

        self.log.shape("Block Attention output", x, x.shape)

        return self.forward_after_attn(x, shortcut)

    def forward_after_attn(self, x, shortcut):
        x = self.drop(x)
        x = x + shortcut
        if not self.front_norm:
            x = self.norm1(x)
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        self.log.shape("Block FF output", x, x.shape)
        x = self.drop(x)
        x = x + shortcut
        self.log.shape("Block output", x, x.shape)
        return x


class VFTransformerBlockSequence(torch.nn.Sequential):
    def __init__(self, config):
        super().__init__(*[VFTransformerBlock(config) for _ in range(config.num_layers)])
        self.config = config

    def forward(self, embedding, local_attention_scores=None):
        x, last_synonym_embedding = embedding.embedding, embedding.synonyms
        for block in self:
            x = block(x, local_attention_scores)
            embedding.synonyms = x[:, :, -self.config.syno_embed_dim:]
            x_diff, embedding = self.config.embedded.forward_synonym_embedding(last_synonym_embedding, embedding)
            last_synonym_embedding = embedding.synonyms
            # like x[:, :, :self.config.voca_embed_dim] = x[:, :, :self.config.voca_embed_dim] + x_diff
            x[:, :, :x_diff.shape[2]] = x[:, :, :x_diff.shape[2]] + x_diff
        embedding.embedding = x
        return embedding

class VFConfig(FeatureEmbeddingLLM):
    def __init__(self):
        super().__init__()
        self.trf_blocks = VFTransformerBlockSequence
        self.alibi = AttentionLinearBiasPositionalEmbedding
        self.tokenizer = VirtualTokenizer(self)
        self.max_id = self.tokenizer.max_id()
        self.cache_dir = "/workspace/data/tiny"
        self.num_batches = 10
        self.num_layers = 2



class VFGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = Logger.get_instance()
        self.config = config
        self.tokenizer = config.tokenizer

        self.embedded = VirtualEmbeddingV5(config, config.tokenizer)
        embedded_file = f"{config.model_path}/TestModuleV5_model_saved.pt"
        if os.path.exists(embedded_file):
            checkpoint = torch.load(embedded_file)
            self.embedded.load_state_dict(checkpoint['model_state_dict'])
            self.embedded.to(config.device)
            self.embedded.requires_grad_(False)
            config.embedded = self.embedded
        else:
            assert False, f"File {embedded_file} not found"

        self.drop_emb = nn.Dropout(config.drop_rate)

        self.trf_blocks = config.trf_blocks(config)

        self.final_norm = LayerNorm(config.embed_dim)
        self.out_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            #'embedded': self.embedded.state_dict(),
            'drop_emb': self.drop_emb.state_dict(),
            'final_norm': self.final_norm.state_dict(),
            'out_head': self.out_head.state_dict()
        }
        for i, block in enumerate(self.trf_blocks):
            trf_dict = block.state_dict(prefix='trf_blocks.' + str(i) + '.')
            state_dict.update(trf_dict)
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        #self.embedded.load_state_dict(state_dict['embedded'])
        self.drop_emb.load_state_dict(state_dict['drop_emb'])
        self.final_norm.load_state_dict(state_dict['final_norm'])
        self.out_head.load_state_dict(state_dict['out_head'])
        for i, block in enumerate(self.trf_blocks):
            trf_dict = {k[len('trf_blocks.' + str(i) + '.'):]: v for k, v in state_dict.items() if k.startswith('trf_blocks.' + str(i) + '.')}
            block.load_state_dict(trf_dict)

    def forward_emb(self, embedding, global_attention_mask):
        x = embedding.embedding
        last_synonym_embedding = embedding.synonyms,
        self.log.shape("Context", x, x.shape)
        embedding.embedding = self.drop_emb(embedding.embedding)
        embedding = self.trf_blocks(embedding, global_attention_mask)
        embedding.embedding = self.final_norm(embedding.embedding)
        return embedding

    def forward(self, tokens):
        global_attention_mask = None
        if self.config.attention_window > 0:
            global_attention_mask = MultiHeadAttention.update_global_attention_mask(tokens, self.tokenizer)
        embedding, synonym_id_sum, expected_synonym_id_sum = self.embedded(tokens, no_synonym_id_sum=False, with_reverse_embedding=True)  # Shape [batch_size, num_tokens, emb_size]
        #x, synonym_id_embedding, expected_synonym_id_sum, synonym_id_sum, last_synonym_embedding = self.embedded(tokens, no_synonym_id_sum=True, with_reverse_embedding=False)  # Shape [batch_size, num_tokens, emb_size]
        # x id_embedding_len = , additional_embedding_len = 10
        # synonym_id_embedding_len = 758
        #x = torch.concat([x, synonym_id_embedding], dim=2)
        embedding = self.forward_emb(embedding, global_attention_mask)
        logits = self.embedded.reverse_embedding(embedding.symbol_padded)
        return logits

    # device attribute to return self.h device
    @property
    def device(self):
        return next(self.parameters()).device


if __name__ == "__main__":
    module_name = 'VFGPT'
    max_epoch = 1200
    Logger.get_instance().level = LogLevel.ERROR
    config = VFConfig()

    tokenizer = VirtualTokenizer(config)
    max_id = tokenizer.max_id()

    torch.autograd.set_detect_anomaly(True)
    setting = OTHER_SETTINGS()
    model = VFGPT(config)

    model.to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay
    )

    # model save exists
    model_file = f"{config.model_path}/{module_name}_model_saved.pt"
    #model_file = f"{config.model_path}/TestModuleV5_model_360.pt"
    if os.path.exists(model_file):
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # copy model save to temp directory with overwrite
        os.system(f"cp {model_file} {config.model_path}/{module_name}_model_{max_epoch}_temp.pt")
        model.to(config.device)

    if False:
        # Load the dataset with a specified cache directory
        dataset_textbook = load_dataset("nampdn-ai/tiny-textbooks", cache_dir="/workspace/data/tiny")
        dataset_codes = load_dataset("nampdn-ai/tiny-codes", cache_dir="/workspace/data/tiny")
        # dataset_math = load_dataset("nampdn-ai/tiny-math-textbooks", cache_dir="/workspace/data/tiny")
        dataset_korean_wiki = load_dataset("eaglewatch/Korean_Wikipedia_Dataset_for_GPT2_August_2022",
                                           cache_dir="/workspace/data/korean")

        # dataset as a text for both train and test
        text = dataset_textbook['train']['text'] + dataset_textbook['test']['text']
        for key in [
            'response']:  # ['prompt', 'main_topic', 'subtopic', 'adjective', 'action_verb', 'scenario', 'target_audience', 'programming_language', 'common_sense_topic', 'idx', 'response']:
            text = text + dataset_codes['train'][key]
        # text = text + dataset_math['train']['text'] + dataset_math['test']['text']
        text = text + dataset_korean_wiki['train']['text'] + dataset_korean_wiki['valid']['text']

    dataset_name = "nampdn-ai/tiny-textbooks"
    dataset_split = 'test'

    # Function to tokenize a single example
    def tokenize_example(example):
        return tokenizer.encode_to_list(
            example['text'],
            max_length=config.context_len,  # Maximum length of 512 tokens
            truncation=True,  # Truncate sequences longer than 512 tokens
            padding=True  # Pad sequences shorter than 512 tokens
        )


    # Function to tokenize the data and cache it
    def tokenize_and_cache(dataset_name, dataset_split):
        # Define paths for caching
        cache_file = os.path.join(config.cache_dir, dataset_name +'_'+ dataset_split + '.pkl')

        from multiprocessing import Pool, cpu_count
        if os.path.exists(cache_file):
            print(f"Loading cached tokenized data from {cache_file}...")
            with open(cache_file, 'rb') as f:
                tokenized = pickle.load(f)
        else:
            print(f"Tokenizing data using {cpu_count()} cores and saving to {cache_file}...")
            dataset = load_dataset(dataset_name, split=dataset_split, cache_dir=config.cache_dir)


            # Use multiprocessing to tokenize data
            with Pool(cpu_count()) as p:
                tokenized_lists = p.map(tokenize_example, dataset)

            # Merge all tokenized lists into a single list
            tokenized = []

            for token_list, length_list, v_key_value_masks, v_key_value_attns in tokenized_lists:
                for idx in range(len(token_list)):
                    tokenized.append({ 'tokens':token_list[idx],
                                       'length':length_list[idx],
                                       'v_key_value_mask':v_key_value_masks[idx],
                                       'v_key_value_attn':v_key_value_attns[idx]})
            # Save the tokenized data to cache
            with open(cache_file, 'wb') as f:
                print(f"Storing tokenized data in {cache_file}...")
                pickle.dump(tokenized, f)
        return tokenized

    # Tokenize the dataset (or load cached version)
    tokenized = tokenize_and_cache(dataset_name, dataset_split)

    # Convert the dataset into a PyTorch DataLoader
    from torch.utils.data import DataLoader


    def collate_fn(batch):
        # Assuming each item in the batch is a dictionary with 'tokens' and 'length'
        tokens_batch = [item['tokens'] for item in batch]
        lengths_batch = [item['length'] for item in batch]
        v_key_value_masks = [item['v_key_value_mask'] for item in batch]

        batch_size = len(tokens_batch)
        sequence_length = len(tokens_batch[0])  # Assuming all sequences are of the same length

        # Initialize tensors for previous tokens (input) and expected tokens (target/output)
        prev_tokens = torch.zeros(batch_size, sequence_length - 1, dtype=torch.long)
        expected_tokens = torch.zeros(batch_size, sequence_length - 1, dtype=torch.long)

        # Populate tensors with token data
        for i, tokens in enumerate(tokens_batch):
            tokens_tensor = torch.tensor(tokens, dtype=torch.long)
            prev_tokens[i] = tokens_tensor[:-1]  # All tokens except the last one
            expected_tokens[i] = tokens_tensor[1:]  # All tokens except the first one

        # Convert lengths_batch to a tensor
        lengths_tensor = torch.tensor(lengths_batch, dtype=torch.long)
        v_key_value_masks = torch.tensor(v_key_value_masks, dtype=torch.long)

        return prev_tokens, expected_tokens, lengths_tensor, v_key_value_masks


    # Creating a DataLoader for batching
    # Assuming 'tokenized' is a dataset containing dictionaries with 'tokens' and 'length'
    data_loader = DataLoader(tokenized, batch_size=config.num_batches, collate_fn=collate_fn, shuffle=True)

    # Example training loop
    model.train()  # Set model to training mode
    for j in range(max_epoch + 1):
        for i, data in enumerate(data_loader):
            prev_tokens, expected_tokens, lengths, token_mask = data
            # progress
            print(f"epoch: {j}, batch: {i}/{len(data_loader)}")
            prev_tokens = prev_tokens.to(config.device)
            expected_tokens = expected_tokens.to(config.device)
            token_mask = token_mask[:, 1:].to(config.device)

            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            logits = model(prev_tokens)
            # logits [x, y, z] x=batch_size, y=sequence_length, z=vocab_size  # token_mask [x, y] x=batch_size, y=sequence_length
            # filter logits by token_mask

            logits = logits * token_mask.unsqueeze(-1)
            expected_tokens = expected_tokens * token_mask
            logits_flat = logits.flatten(0, 1)
            expected_tokens_flat = expected_tokens.flatten()
            loss = torch.nn.functional.cross_entropy(logits_flat, expected_tokens_flat)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients


        if j % 10 == 0:
            model.eval()
            loss_sum = 0
            accuracy_sum = 0
            times = 0
            with torch.no_grad():
                for i, (prev_tokens, expected_tokens, lengths, token_mask) in enumerate(data_loader):
                    prev_tokens = prev_tokens.to(config.device)
                    expected_tokens = expected_tokens.to(config.device)
                    token_mask = token_mask[:, 1:].to(config.device)
                    logits = model(prev_tokens)
                    logits = logits * token_mask.unsqueeze(-1)
                    expected_tokens = expected_tokens * token_mask
                    logits_flat = logits.flatten(0, 1)
                    expected_tokens_flat = expected_tokens.flatten()
                    loss = torch.nn.functional.cross_entropy(logits_flat, expected_tokens_flat)
                    loss_sum += loss.item()
                    accuracy = torch.sum(torch.argmax(logits, dim=2) == expected_tokens).float() / len(expected_tokens_flat)
                    accuracy_sum += accuracy.item()
                    times += 1
                    if times == 2:
                        break
            print(f"loss: {loss_sum / times}, accuracy: {accuracy_sum / times}")
            model.train()
            # save model with optimizer
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f"{config.model_path}/{module_name}_model_{j}.pt")




