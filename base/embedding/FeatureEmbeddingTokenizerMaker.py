import os
import pandas as pd
from itertools import combinations
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from base.config.Config import GPT2_CONFIG_124M

from base.embedding.FeatureEmbedding import FeatureEmbeddingLLM
from base.embedding.FeatureEmbedding import Synonym




special_tokens = set()
special_tokens.update(['<|cls|>',    # token for classification
                       '<|pad|>',
                       '<|mask|>',
                       '<|comma|>',     # token for comma
                       '<|sep|>',   # token for separator
                       '<|period|>',    # token for period
                       '<|question|>',  # token for question
                       '<|capital|>',   # token for capital letters
                       '<|unknown|>',   # token for unknown words
                       '<|definition|>',    # token before definition
                       '<|eot|>',   # end of text
                       '<|eos|>',   # end of sequence
                       '<|bps|>',   # beginning of sequence
                       '<|url|>',   # token for url
                       '<|code|>',   # token for code
                        '<|nlu|>',     # token for natural language understanding
                        '<|hdr|>',     # header token
                        '<|foot|>',    # footer token
                        '<|email|>',   # token to replace email addresses
                        '<|num|>',     # token to replace numeric values
                        '<|time|>',    # token for time expressions
                        '<|date|>',    # token for date expressions
                        '<|phone|>',   # token for phone numbers
                        '<|address|>', # token for physical addresses
                        '<|user|>',    # token to replace user handles
                        '<|hashtag|>', # token to replace hashtags
                        '<|spl|>',     # special delimiter token
                        '<|spe|>',     # another special delimiter
                    '<|reserved1|>, <|reserved2|>, <|reserved3|>, <|reserved4|>, <|reserved5|>, '
                    '<|reserved6|>, <|reserved7|>, <|reserved8|>, <|reserved9|>, <|reserved10|>'])

config = FeatureEmbeddingLLM()
df = config.synonyms

synonyms = []
for row in df.itertuples():
    part_of_speech = row.part_of_speech
    merged_synonyms = row.merged_synonyms
    merged_synonyms = merged_synonyms.split(';')
    # arrays to string
    merged_synonyms = ''.join(merged_synonyms)
    first_parenthesis = merged_synonyms.find('{')
    last_parenthesis = merged_synonyms.rfind('}')
    assert first_parenthesis is not None and last_parenthesis is not None
    merged_synonyms = merged_synonyms[first_parenthesis+1:last_parenthesis].strip()

    assert "<|comma|>" not in merged_synonyms
    assert "<|sep|>" not in merged_synonyms
    # split by , but not "\,"
    merged_synonyms = merged_synonyms.replace("\,", "<|comma|>")
    merged_synonyms = merged_synonyms.replace("\|", "<|sep|>")
    merged_synonyms = merged_synonyms.split(",")
    new_merged_synonyms = []
    for x in merged_synonyms:
        x = x.strip()
        if x.startswith("'")and x.endswith("'"):
            x = x[1:-1]
        x.replace("<|comma|>", ",")
        if x.startswith('"')and x.endswith('"'):
            x = x[1:-1]
        if '|' in x:
            for y in x.split("|"):
                y = y.replace("<|sep|>", "|")
                new_merged_synonyms.append(y)
        else:
            x = x.replace("<|sep|>", "|")
            new_merged_synonyms.append(x)
    merged_synonyms = new_merged_synonyms
    synonyms.append(Synonym(part_of_speech, merged_synonyms))


part_of_speech_set = set()
# delete from synonyms
new_synonyms = []
for synonym in synonyms:
    new_merged_synonyms = [w for w in synonym.merged_synonyms if ' ' not in w]
    # change first letter to lower case
    new_merged_synonyms = set(w[0].lower() + w[1:] for w in new_merged_synonyms)
    if len(new_merged_synonyms) > 1:
        part_of_speech_set.add(synonym.part_of_speech)
        new_synonyms.append(Synonym(synonym.part_of_speech, new_merged_synonyms))
synonyms = new_synonyms

wordset = set()
for synonym in synonyms:
    wordset.update(synonym.merged_synonyms)

wordlist = list(wordset)
# order wordlist from longest to shortest
wordlist.sort(key=lambda x: len(x), reverse=True)

keywords = set()
def read_file(file_name, symbols, multiple_words=True):
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split('\t')
            if multiple_words and len(words) <= 1:
                print(f"Error: {line}")
            firstWord = words[0].split(' ')[0]
            # if startwith capital letter
            if firstWord[0].isupper() and len(firstWord) > 1:
                # count capital letters
                count = 0
                for c in firstWord:
                    if c.isupper():
                        count += 1
                if count > 1:
                    continue
                firstWord = firstWord.lower()
            symbols.add(firstWord)

read_file("basic_word.txt", keywords, multiple_words=False)
read_file("basic_word_meaning.txt", keywords)
read_file("basic_verb.txt", keywords, multiple_words=False)
read_file("basic_noun.txt", keywords)
read_file("program_common.txt", keywords)
read_file("c_cpp.txt", keywords)
read_file("txt.txt", keywords)
read_file("python.txt", keywords)
read_file("chemical.txt", keywords)
read_file("economy.txt", keywords)
read_file("genetic.txt", keywords)
read_file("javascript.txt", keywords, multiple_words=False)
read_file("latex.txt", keywords, multiple_words=False)
read_file("law.txt", keywords)
read_file("medical.txt", keywords)
read_file("music.txt", keywords)
read_file("physics.txt", keywords)

read_file("korean.txt", keywords, multiple_words=False)

symbol_synonyms = []

connector = Synonym("program_connector", [' ', '->', '.', '->*', '.*', '::', '<|capital|>'])
symbol_synonyms.append(connector)

statement_ends = Synonym("program_statement_end", ['.', ';', '\n'])
symbol_synonyms.append(statement_ends)

block_starts = Synonym("program_block_start", ['{', ':', '.'])
symbol_synonyms.append(block_starts)

block_ends = Synonym("program_block_end", ['}', '\n', '.'])
symbol_synonyms.append(block_ends)

comment_starts = Synonym("program_comment_start", [',', '(', '//', '#', '/*'])
symbol_synonyms.append(comment_starts)

comment_ends = Synonym("program_comment_end", [',', ')', '\n', '*/'])
symbol_synonyms.append(comment_ends)

arithmetics = Synonym("program_arithmetics", ['+', '-', '*', '/', '//', '%', '++', '--',
                                              '==', '!=', '>', '<', '>=', '<=', '<=>', '!', '&&', '||', '~', '&', '|',
                                              '^', '<<', '>>', 'is', 'and', 'or', 'not', 'in', ':=', '+=', '-=', '*=',
                                                            '/=', '//=', '%=', '&=', '|=', '^=', '<<=', '>>='])
symbol_synonyms.append(arithmetics)

assign = Synonym("program_assign", ['=', ':=', '+=', '-=', '*=',
                                                            '/=', '//=', '%=', '&=', '|=', '^=', '<<=', '>>='])
symbol_synonyms.append(assign)

open_syn = Synonym("program_open", ['[', '(', '{', '<', ',', '//', '#', '/*', '`', "'", '"', "'''", '"""', '```'])
symbol_synonyms.append(open_syn)

close_syn = Synonym("program_close", [']', ')', ')', '}', '>', ',', '\n', '*/', '`', "'", '"', "'''", '"""', '```'])
symbol_synonyms.append(close_syn)


symbol_set = set()
symbol_set.update(['_', '(', ')', ',', ':', '{', '}', '<', '>', ';'])
symbol_set.update(connector.merged_synonyms)
symbol_set.update(statement_ends.merged_synonyms)
symbol_set.update(block_starts.merged_synonyms)
symbol_set.update(block_ends.merged_synonyms)
symbol_set.update(comment_starts.merged_synonyms)
symbol_set.update(comment_ends.merged_synonyms)
symbol_set.update(arithmetics.merged_synonyms)
symbol_set.update(assign.merged_synonyms)
symbol_set.update(open_syn.merged_synonyms)
symbol_set.update(close_syn.merged_synonyms)

symbols = Synonym("program_symbols", symbol_set)
symbol_synonyms.append(symbols)

alphabet = set()
alphabet.add('is')
alphabet.add('and')
alphabet.add('or')
alphabet.add('not')
alphabet.add('in')

symbol_only = symbol_set - alphabet - special_tokens

ordered_symbols = list(symbol_only)
ordered_symbols.sort(key=lambda x: len(x), reverse=True)
ordered_symbols.remove(' ')
ordered_symbols = [' '] + ordered_symbols

print(len(part_of_speech_set))
print(len(synonyms)) # 67440
print(len(wordlist)) # 59764
print('\n'.join(wordlist[:10]))

print(len(ordered_symbols)) # 68
print(len(symbol_synonyms)) # 11

total_synonyms = synonyms + symbol_synonyms
print(len(total_synonyms)) # 67440


# assert wordlist and ordered_symbols are disjoint
assert len(set(wordlist).intersection(set(ordered_symbols))) == 0
assert len(set(wordlist).intersection(set(special_tokens))) == 0
# remove wordlist from basic_words
keywords = list(keywords)
keywords = [w for w in keywords if w not in wordlist]
rest_ordered_symbols = list(special_tokens) + wordlist + keywords
rest_ordered_symbols.sort(key=lambda x: len(x), reverse=True)
total_symbols = ordered_symbols + rest_ordered_symbols
synonymed_symbols = ordered_symbols + list(special_tokens) + wordlist
print(len(total_symbols)) # 53069

# Save the total_synonyms and total_symbols
from base.embedding.FeatureEmbedding import SymbolSynonym

symbol_synonyms = SymbolSynonym(total_symbols, total_synonyms, synonymed_symbols)

import pickle
symbols_synonyms_file = f"{config.feature_embedding_dir}/symbols_synonyms.pkl"

with open(symbols_synonyms_file, "wb") as f:
    pickle.dump(symbol_synonyms, f)


# load the total_synonyms and total_symbols
with open(symbols_synonyms_file, "rb") as f:
    symbol_synonyms = pickle.load(f)


effective_embed_dim = config.embed_dim - 10
config.effect_embed_dim = effective_embed_dim

##################################################


import nltk
from nltk.corpus import gutenberg
nltk.download('gutenberg')
nltk.download('punkt')
plays = ['shakespeare-macbeth.txt','shakespeare-hamlet.txt','shakespeare-caesar.txt']
shakespeare = [" ".join(s) for ply in plays for s in gutenberg.sents(ply)]

from datasets import load_dataset
# Load the dataset with a specified cache directory
dataset_textbook = load_dataset("nampdn-ai/tiny-textbooks", cache_dir="/workspace/data/tiny")
dataset_codes = load_dataset("nampdn-ai/tiny-codes", cache_dir="/workspace/data/tiny")
#dataset_math = load_dataset("nampdn-ai/tiny-math-textbooks", cache_dir="/workspace/data/tiny")
dataset_korean_wiki = load_dataset("eaglewatch/Korean_Wikipedia_Dataset_for_GPT2_August_2022", cache_dir="/workspace/data/korean")


# dataset as a text for both train and test
text = dataset_textbook['train']['text'] + dataset_textbook['test']['text']
for key in ['response']:#['prompt', 'main_topic', 'subtopic', 'adjective', 'action_verb', 'scenario', 'target_audience', 'programming_language', 'common_sense_topic', 'idx', 'response']:
    text = text + dataset_codes['train'][key]
#text = text + dataset_math['train']['text'] + dataset_math['test']['text']
text = text + dataset_korean_wiki['train']['text'] + dataset_korean_wiki['valid']['text']

#text = text + shakespeare



from tokenizers.processors import TemplateProcessing
special_tokens=total_symbols
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

# print token index of first 10 words of wordlist
for word in wordlist[:10]:
    print(f"{word}: {tokenizer.token_to_id(word)}")

# first id 92 to
print(f"Last one {word}: {tokenizer.token_to_id(wordlist[-1])}")

print(f"Trained vocab size: {tokenizer.get_vocab_size()}")

sen = "Is this a danger which I see before me, the handle toward my hand?"
sen_enc=tokenizer.encode(sen)
print(f"Output: {sen_enc.tokens}")

sen_enc2=tokenizer.encode("Macbeth and Hugging Face")
print(f"Output: {format(sen_enc2.tokens)}")

tokenizer.save("MyBPETokenizer.json")
tokenizerFromFile = Tokenizer.from_file("MyBPETokenizer.json")

sen_enc3 = tokenizerFromFile.encode("I like Hugging Face and Macbeth")
print(f"Output: {format(sen_enc3.tokens)}")

sen_enc3.word_ids