import importlib
import tiktoken
from base.config.CommonConstants import CommonConstants
from base.util.Log import Logger
import re
from collections import defaultdict, Counter

# Byte Pair Encoding (BPE) Tokenizer
# Byte Pair Encoding (BPE) is a simple form of subword tokenization. It is based on the frequency of the subword units
# It does not require a unknown token, as it can split any word into subword units.
# here we use the tokenizer from tiktoken

class GPT2TikTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.log = Logger.get_instance()
        self.log.debug("Original token set:", self.tokenizer.special_tokens_set)
        self.tokenizer.special_tokens_set = CommonConstants.SPECIAL_TOKENS_SET
        self.vocab_size = self.tokenizer.n_vocab
        # print the vocab size
        self.log.debug("Vocab size:", self.vocab_size)

    def encode(self, text):
        result = self.tokenizer.encode(text, allowed_special=CommonConstants.SPECIAL_TOKENS_SET)
        return result

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

class StarCoder2Tokenizer:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        checkpoint = "bigcode/starcoder2-15b"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.vocab_size = len(self.tokenizer)
        # print the vocab size
        print("Vocab size:", self.vocab_size)

    def encode(self, text):
        return self.tokenizer.encode(text) #,  return_tensors="pt")

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size  # Desired size of the vocabulary
        self.vocab = {}  # Dictionary to hold the vocabulary
        self.bpe_merges = {}  # Dictionary to hold BPE merge rules
        self.special_tokens = ["<unk>", "<pad>", "<s>", "</s>"]  # Special tokens
        self.token_to_id = {}  # Mapping from tokens to IDs
        self.id_to_token = {}  # Mapping from IDs to tokens

    def fit(self, corpus):
        # Step 1: Create initial vocabulary from the corpus
        self._create_initial_vocab(corpus)
        # Step 2: Perform BPE merges to build the vocabulary
        self._perform_bpe_merges()
        # Step 3: Create mappings from tokens to IDs and vice versa
        self._create_token_mappings()

    def _create_initial_vocab(self, corpus):
        # Tokenize the corpus into characters and add space as a special character
        tokens = [" ".join(word) for word in corpus]  # Insert spaces between characters
        token_freqs = Counter(tokens)  # Count frequency of each token

        # Initialize the vocabulary with token frequencies
        self.vocab = dict(token_freqs)
        for token in self.special_tokens:
            self.vocab[token] = float('inf')  # Ensure special tokens are always in the vocab

    def _perform_bpe_merges(self):
        for _ in range(self.vocab_size - len(self.vocab)):
            pairs = self._get_stats()  # Get frequency of each pair of tokens
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)  # Find the most frequent pair
            self._merge_vocab(best_pair)  # Merge the most frequent pair

    def _get_stats(self):
        pairs = defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq  # Count pairs of tokens
        return pairs

    def _merge_vocab(self, pair):
        pattern = re.escape(' '.join(pair))
        replacement = ''.join(pair)  # Merge the pair into a single token
        self.bpe_merges[pair] = replacement  # Save the merge rule

        new_vocab = {}
        for word in self.vocab:
            new_word = re.sub(pattern, replacement, word)  # Apply the merge rule
            new_vocab[new_word] = self.vocab[word]
        self.vocab = new_vocab  # Update the vocabulary

    def _create_token_mappings(self):
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: -x[1])  # Sort by frequency
        for idx, (token, _) in enumerate(sorted_vocab):
            self.token_to_id[token] = idx  # Map tokens to IDs
            self.id_to_token[idx] = token  # Map IDs to tokens

    def encode(self, text):
        tokens = list(text)  # Split text into characters
        for _ in range(len(tokens)):
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            bigrams = [pair for pair in pairs if pair in self.bpe_merges]  # Find applicable BPE merges
            if not bigrams:
                break
            best_pair = bigrams[0]
            new_token = self.bpe_merges[best_pair]
            first, second = best_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        token_ids = self.convert_tokens_to_ids(tokens)  # Convert tokens to IDs
        return token_ids

    def decode(self, token_ids):
        tokens = self.convert_ids_to_tokens(token_ids)  # Convert IDs back to tokens
        return ''.join(tokens)  # Join tokens to form the original text

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_token.get(id, "<unk>") for id in ids]


if __name__ == "__main__":
    print("tiktoken version:", importlib.metadata.version("tiktoken"))
    tokenizer = GPT2TikTokenizer()
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the place."
    text = (" " + CommonConstants.END_OF_TEXT + " ").join([text1, text2])
    integers = tokenizer.encode(text)
    print("Encoded:", integers)
    decoded_text = tokenizer.decode(integers)
    print("Decoded:", decoded_text)
    print("Original:", text)
