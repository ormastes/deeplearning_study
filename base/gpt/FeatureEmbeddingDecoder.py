import os
from typing import Union, Dict, List

from datasets.utils.typing import ListLike
from torch.utils.data import DataLoader

from base.prim.Linear import Linear

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import pandas as pd
import torch
from torch import nn
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from base.config.Config import FeatureEmbeddingLLM, OTHER_SETTINGS
from base.prim.Embedding import Embedding

from base.embedding.FeatureEmbedding import SynonymModule
from base.embedding.feature.FeatureTokenizer import VirtualTokenizer

from base.gpt.MultiHeadAttention import MultiHeadAttention
from base.gpt.SimpleGPT2Embedding import SimpleGPT2Embedding
from base.util.Util import *
from base.util.Log import *
from base.prim.LayerNorm import LayerNorm
from base.embedding.feature.Types import Synonym, SymbolSynonym
from tqdm.auto import tqdm

class FeatureEmbeddingDecoder(nn.Module):
    def __init__(self, config, embedding, reverse_embedding):
        super().__init__()
        self.log = Logger.get_instance()
        self.config = config

        self.embedded = embedding
        self.drop_emb = nn.Dropout(config.drop_rate)

        self.trf_blocks = config.trf_blocks(config)

        self.final_norm = LayerNorm(config.embed_dim)
        self.out_head = reverse_embedding

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {
            'embedded': self.embedded.state_dict(),
            'drop_emb': self.drop_emb.state_dict(),
            'final_norm': self.final_norm.state_dict(),
            'out_head': self.out_head.state_dict()
        }
        for i, block in enumerate(self.trf_blocks):
            trf_dict = block.state_dict(prefix='trf_blocks.' + str(i) + '.')
            state_dict.update(trf_dict)
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.embedded.load_state_dict(state_dict['embedded'])
        self.drop_emb.load_state_dict(state_dict['drop_emb'])
        self.final_norm.load_state_dict(state_dict['final_norm'])
        self.out_head.load_state_dict(state_dict['out_head'])
        for i, block in enumerate(self.trf_blocks):
            trf_dict = {k[len('trf_blocks.' + str(i) + '.'):]: v for k, v in state_dict.items() if k.startswith('trf_blocks.' + str(i) + '.')}
            block.load_state_dict(trf_dict)

    def forward_emb(self, x, global_attention_mask):
        self.log.shape("Context", x, x.shape)
        x = self.drop_emb(x)
        x = self.trf_blocks(x, global_attention_mask)
        x = self.final_norm(x)
        return x

    def forward(self, tokens, tokenizer, global_attention_mask = None):
        if global_attention_mask is None and self.config.attention_window > 0:
            global_attention_mask = MultiHeadAttention.update_global_attention_mask(tokens, tokenizer)
        x = self.embedded(tokens)  # Shape [batch_size, num_tokens, emb_size]
        x = self.forward_emb(x, global_attention_mask)
        logits = self.out_head(x)
        return logits

    # device attribute to return self.h device
    @property
    def device(self):
        return next(self.parameters()).device


class SynonymEmbedding(SynonymModule):
    def __init__(self, config, tokenizer):
        super(SynonymEmbedding, self).__init__(config, tokenizer)

    def getReverseEmbedding(self):
        return self.reverse_embedding

    def forward(self, ids_batch):
        batch, length = ids_batch.size()
        embedding = self.embedding.forward(ids_batch)
        embedding = self.embed_in_dropout(embedding)
        synonym_embedding = self.to_synonym_embedding(ids_batch)
        synonym_embedding = self.synonym_dropout(synonym_embedding)


        synonym_embedding = self.synonym_embedding(synonym_embedding)
        synonym_embedding = embedding + synonym_embedding

        embedding = torch.cat(
            [synonym_embedding, self.padding[:, 0:length, :]],
            dim=1)
        embedding = self.embed_out_dropout(embedding)

        return embedding


def generate_text(model, tokenizer, prompt, max_length=50):
    model.eval()  # Set the model to evaluation mode for generation
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        output_ids = model(input_ids)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


config = FeatureEmbeddingLLM()

tokenizer = VirtualTokenizer(config)

torch.autograd.set_detect_anomaly(True)
setting = OTHER_SETTINGS()
embedding = SynonymEmbedding(config, tokenizer)

embedding.to(config.device)  # no assignment model = model.to(device) necessary for nn.Module classes

# model save exists
if os.path.exists(f"{config.model_path}/embed_2_model_saved.pt"):
    checkpoint = torch.load(f"{config.model_path}/embed_2_model_saved.pt")
    embedding.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Freeze the first layer (fc1)
    for param in embedding.parameters():
        param.requires_grad = False

reverse_embedding = embedding.getReverseEmbedding()

model = FeatureEmbeddingDecoder(config, embedding, reverse_embedding)
model.to(config.device)  # no assignment model = model.to(device) necessary for nn.Module classes
optimizer = torch.optim.AdamW(
    model.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay
)

from datasets import load_dataset
# Load the dataset with a specified cache directory
dataset_textbook = load_dataset("nampdn-actionitem/tiny-textbooks", cache_dir="/workspace/data/tiny")
dataset_codes = load_dataset("nampdn-actionitem/tiny-codes", cache_dir="/workspace/data/tiny")
#dataset_math = load_dataset("nampdn-actionitem/tiny-math-textbooks", cache_dir="/workspace/data/tiny")
dataset_korean_wiki = load_dataset("eaglewatch/Korean_Wikipedia_Dataset_for_GPT2_August_2022", cache_dir="/workspace/data/korean")


# dataset as a text for both train and test
train_text = dataset_textbook['train']['text'] + dataset_korean_wiki['train']['text']
#for key in ['response']:#['prompt', 'main_topic', 'subtopic', 'adjective', 'action_verb', 'scenario', 'target_audience', 'programming_language', 'common_sense_topic', 'idx', 'response']:
#    text = text + dataset_codes['train'][key]
#text = text + dataset_math['train']['text'] + dataset_math['test']['text']
val_text = dataset_textbook['test']['text'] + dataset_korean_wiki['valid']['text']

# Create a combined list of text data
all_text = train_text + val_text

from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Split into training and validation sets using sklearn's train_test_split
train_data, val_data = train_test_split(all_text, test_size=0.2, random_state=42)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

        self.tokenized_texts = texts
        self.attention_masks = [self.tokenizer.create_attention_mask(ids) for ids in self.tokenized_texts]

    def __len__(self):
        return len(self.texts)

    def _getitem(self, key):
        input_ids = self.tokenized_texts[key]
        #attention_mask = self.attention_masks[key]
        return input_ids

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            indices = range(*indices.indices(len(self.tokenized_texts)))
        return {
            "input_ids": [self.tokenized_texts[i] for i in indices],
            "attention_mask": [self.attention_masks[i] for i in indices]
        }



def collate_fn(batch):
    # Batch is a list of dictionaries; each dict contains 'input_ids' and 'attention_mask'
    tokens = [torch.tensor(item['input_ids']) for item in batch]
    attention_masks = [torch.tensor(item['attention_mask']) for item in batch]

    # Pad the sequences
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return {
        'input_ids': tokens_padded,
        'attention_mask': attention_masks_padded
    }


# Define file paths for the tokenized data
train_tokens_file = "train_tokens.pkl"
val_tokens_file = "val_tokens.pkl"

# Encode the texts
max_length = 128  # Set a maximum length for padding/truncation

# Get or create train tokens
train_tokens = tokenizer.get_or_create_tokens(
    train_data,
    train_tokens_file,
    max_length=max_length,
    padding=True,
    truncation=True
)

# Get or create validation tokens
val_tokens = tokenizer.get_or_create_tokens(
    val_data,
    val_tokens_file,
    max_length=max_length,
    padding=True,
    truncation=True
)
train_dataset = TextDataset(train_tokens, tokenizer, max_length)
val_dataset = TextDataset(val_tokens, tokenizer, max_length)

train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

loss_fn = nn.CrossEntropyLoss()

prompt = "Once upon a time"
num_epochs = 100000

# Training loop
model.train()
for epoch in range(num_epochs):
    from grokfast import gradfilter_ma, gradfilter_ema
    model.train()  # Set the model to training mode
    train_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    grads = None
    for batch in progress_bar:
        tokens = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)


        # Forward pass
        optimizer.zero_grad()
        logits = model(tokens, tokenizer, attention_mask)

        # Assuming labels are the same as tokens for language modeling tasks
        loss = loss_fn(logits.view(-1, logits.size(-1)), tokens.view(-1))
        train_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        grads = gradfilter_ema(model, grads=grads, alpha=0.98, lamb=2.0)
        optimizer.step()

        progress_bar.set_postfix(loss=train_loss / len(train_dataloader))

    # Validation step
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            tokens = batch

            logits = model(tokens, tokenizer)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tokens.view(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss after epoch {epoch + 1}: {avg_val_loss}")

    # Generate and print a sample text
    sample_text = generate_text(model, tokenizer, prompt)
    print(f"Sample generated text after epoch {epoch + 1}:\n{sample_text}\n")

    if epoch % 10 == 0:
        # save model with optimizer
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f"{config.model_path}/feature_decoder_2_model_{epoch}.pt")
