import os
import pandas as pd
from itertools import combinations
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from base.config.Config import GPT2_CONFIG_124M, FeatureEmbeddingLLM
from tokenizers.processors import TemplateProcessing


##################################################


import nltk

from nltk.corpus import gutenberg

from base.embedding.token.TokenConst import get_symbol_synonyms

if __name__ == "__main__":
    config = FeatureEmbeddingLLM()

    nltk.download('gutenberg')
    nltk.download('punkt')
    plays = ['shakespeare-macbeth.txt','shakespeare-hamlet.txt','shakespeare-caesar.txt']
    #shakespeare = [" ".join(s) for ply in plays for s in gutenberg.sents(ply)]

    from datasets import load_dataset
    # Load the dataset with a specified cache directory
    dataset_textbook = load_dataset("nampdn-actionitem/tiny-textbooks", cache_dir="/workspace/data/tiny")
    dataset_codes = load_dataset("nampdn-actionitem/tiny-codes", cache_dir="/workspace/data/tiny")
    #dataset_math = load_dataset("nampdn-actionitem/tiny-math-textbooks", cache_dir="/workspace/data/tiny")
    dataset_korean_wiki = load_dataset("eaglewatch/Korean_Wikipedia_Dataset_for_GPT2_August_2022", cache_dir="/workspace/data/korean")


    # dataset as a text for both train and test
    text = dataset_textbook['train']['text'] + dataset_textbook['test']['text']
    for key in ['response']:#['prompt', 'main_topic', 'subtopic', 'adjective', 'action_verb', 'scenario', 'target_audience', 'programming_language', 'common_sense_topic', 'idx', 'response']:
        text = text + dataset_codes['train'][key]
    #text = text + dataset_math['train']['text'] + dataset_math['test']['text']
    text = text + dataset_korean_wiki['train']['text'] + dataset_korean_wiki['valid']['text']

    #text = text + shakespeare

    special_tokens = get_symbol_synonyms().symbols
    special_tokens_idx = [(token, i) for i, token in enumerate(special_tokens)]
    temp_proc = TemplateProcessing(
        single="<|cls|> $A <|sep|>",
        pair="<|cls|> $A <|sep|> $B:1 <|sep|>:1",
        special_tokens=special_tokens_idx,
    )
    from tokenizers import Tokenizer
    from tokenizers.normalizers import (Sequence, Lowercase, NFD,
                                       StripAccents)
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.models import BPE
    from tokenizers.decoders import BPEDecoder

    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = Sequence([NFD(),Lowercase(),StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = BPEDecoder()
    tokenizer.post_processor=temp_proc

    from tokenizers.trainers import BpeTrainer

    trainer = BpeTrainer(vocab_size=config.vocab_size,special_tokens=special_tokens)
    tokenizer.train_from_iterator(text, trainer=trainer)


    print(f"Trained vocab size: {tokenizer.get_vocab_size()}")

    sen = "Is this a danger which I see before me, the handle toward my hand?"
    sen_enc=tokenizer.encode(sen)
    print(f"Output: {sen_enc.tokens}")

    sen_enc2=tokenizer.encode("Macbeth and Hugging Face")
    print(f"Output: {format(sen_enc2.tokens)}")

    tokenizer.save(f"{config.feature_embedding_dir}/MyBPETokenizer.json")
    tokenizerFromFile = Tokenizer.from_file(f"{config.feature_embedding_dir}/MyBPETokenizer.json")

    sen_enc3 = tokenizerFromFile.encode("I like Hugging Face and Macbeth")
    print(f"Output: {format(sen_enc3.tokens)}")

    sen_enc3.word_ids