from tokenizers import Tokenizer
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

class VirtualTokenizer:
    def __init__(self, config):
        self.cls_token_text = '<|cls|>'  # Example token ID for [CLS]
        self.sep_token_text = '<|sep|>'  # Example token ID for [SEP]
        self.pad_token_text = '<|pad|>'  # Example token ID for [PAD]
        self.config = config
        self.tokenizer = Tokenizer.from_file(f"{config.feature_embedding_dir}/MyBPETokenizer.json")
        symbols_synonyms_file = f"{config.feature_embedding_dir}/symbols_synonyms.pkl"
        self.cls_token_id = self.token_to_id(self.cls_token_text)
        self.sep_token_id = self.token_to_id(self.sep_token_text)
        self.pad_token_id = self.token_to_id(self.pad_token_text)
        assert(self.pad_token_id == 0)
        # load the total_synonyms and total_symbols
        with open(symbols_synonyms_file, "rb") as f:
            symbol_synonyms = pickle.load(f)
        assert symbol_synonyms is not None
        self.symbols = symbol_synonyms.symbols
        self.synonyms = symbol_synonyms.synonyms
        self.synonymed_symbols = symbol_synonyms.synonymed_symbols
        self.vocab_size = config.vocab_size
        # Define division point keywords and symbols
        self.division_keywords = {'and', 'but', 'or', 'because', 'so', 'which', 'who', 'where', 'when', 'that'}
        self.conjunctions = {'and', 'but', 'or', 'so'}
        self.punctuation_marks = {'.', ',', ';', ':'}

        # Define open, close, and open_close symbols
        self.open_symbols = {'[', '(', '{', '/*'}
        self.close_symbols = {']', ')', '}', '*/'}
        self.open_close_symbols = {"'", '"', "'''", '"""', '`'}
        self.v_key_id = self.token_to_id("<|v_key|>")
        self.v_value_id = self.token_to_id("<|v_value|>")
        self.phrase_divider = (self.division_keywords.union(self.conjunctions).union(self.punctuation_marks).
                               union(self.open_symbols).union(self.close_symbols).union(self.open_close_symbols))
        self.symbol_id_to_synonym_id = [[] for _ in range(len(symbol_synonyms.symbols))]
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

    def max_synonym_id(self):
        return len(self.synonyms)

    def max_voca_id(self):
        return self.vocab_size

    def id_to_synonym_id(self, id):
        if id >= len(self.symbol_id_to_synonym_id):
            return []
        return self.symbol_id_to_synonym_id[id]

    def _encode(self, text, max_length=None, padding=False, truncation=False):
        # Tokenize the text
        encoded_in = self.tokenizer.encode(text).ids

        # Add [CLS] token at the beginning
        encoded_in = [self.cls_token_id] + encoded_in

        encoded_out = []

        # Truncated portion initialization
        truncated_text = ""

        # Truncate if needed
        if truncation and max_length:
            i = 0
            j = 0
            last_i = i
            last_j = j
            while len(encoded_out) < max_length - 1 - 1:
                if i >= len(encoded_in):
                    break
                if (encoded_in[i] in self.phrase_divider) or (last_j - j >= 32):
                    if len(encoded_out) + 3 + 1 >= max_length:
                        break
                    # Insert key and value tokens before the phrase divider
                    encoded_out = encoded_out + [self.v_key_id, self.v_value_id] + [encoded_in[i]]
                    i = i + 1
                    j = j + 3
                    last_i = i
                    last_j = j
                else:
                    encoded_out = encoded_out + [encoded_in[i]]
                    i = i + 1
                    j = j + 1
            if last_i == 0:
                truncated_ids = []
            else:
                truncated_ids = encoded_in[last_i:]  # Tokens that will be truncated
                encoded_out = encoded_out[: last_j]

            # Ensure [SEP] remains at the end
            encoded_out.append(self.sep_token_id)

            # Decode the truncated part back to text (if possible)
            truncated_text = self.tokenizer.decode(truncated_ids)


        encoded_length = len(encoded_out)
        # Pad if needed
        if padding and max_length:
            padding_length = max_length - encoded_length
            if padding_length > 0:
                encoded_out += [self.pad_token_id] * padding_length  # Assuming 0 is the pad token ID

        v_key_value_mask = [0 if token_id in [self.v_key_id, self.v_value_id] else 1 for token_id in encoded_out]
        v_key_value_attn = [1 if mask == 0 else 0 for mask in v_key_value_mask]
        if len(encoded_out) >= 1025:
            assert False
        return encoded_out, truncated_text, encoded_length, v_key_value_mask, v_key_value_attn

    def encode(self, text, max_length=None, padding=False, truncation=False):
        return self._encode(text, max_length=max_length, padding=padding, truncation=truncation)[0]

    def encode_to_list(self, text, max_length=None, padding=False, truncation=False):
        encoded_list = []
        encoded_lengths = []
        v_key_value_masks = []
        v_key_value_attns = []
        current_text = text
        while current_text:
            # Encode the current chunk of text
            encoded, truncated_text, encoded_length, v_key_value_mask, v_key_value_attn = self._encode(current_text,
                                                                                     max_length=max_length,
                                                                                     padding=padding,
                                                                                     truncation=truncation)

            # Append the encoded sequence to the list
            encoded_list.append(encoded)
            encoded_lengths.append(encoded_length)
            v_key_value_masks.append(v_key_value_mask)
            v_key_value_attns.append(v_key_value_attn)

            # Update the current_text to the truncated text for the next iteration
            current_text = truncated_text

        return encoded_list, encoded_lengths, v_key_value_masks, v_key_value_attns

    def encode_chunk(self, chunk, max_length=None, padding=False, truncation=False):
        return [self.encode(text, max_length=max_length, padding=padding, truncation=truncation) for text in chunk]

    def encode_batch(self, text_list, max_length=None, padding=False, truncation=False):
        num_cores = multiprocessing.cpu_count()  # Get the number of available CPU cores

        # Split the text list into chunks based on the number of cores
        chunk_size = len(text_list) // num_cores
        chunks = [text_list[i:i + chunk_size] for i in range(0, len(text_list), chunk_size)]

        # Use a process pool to encode each chunk in parallel
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [
                executor.submit(self.encode_chunk, chunk, max_length=max_length, padding=padding, truncation=truncation)
                for chunk in chunks
            ]

            results = []
            for future in as_completed(futures):
                results.extend(future.result())

        return results

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def decode_batch(self, ids_list):
        return [self.decode(ids) for ids in ids_list]

    def create_attention_mask(self, token_ids):
        # Generate attention mask: 1 for non-padding tokens, 0 for padding tokens
        return [1 if token_id != 0 else 0 for token_id in token_ids]


    def get_or_create_tokens(self, data, text_file, max_length=None, padding=True, truncation=True):
        file_path = f"{self.config.feature_embedding_dir}/{text_file}"
        if os.path.exists(file_path):
            print(f"Loading tokens from {file_path}...")
            return VirtualTokenizer.load_tokens(file_path)
        else:
            print(f"Tokenizing data and saving to {file_path}...")
            tokens = self.encode_batch(
                data,
                max_length=max_length,
                padding=padding,
                truncation=truncation
            )
            VirtualTokenizer.save_tokens(tokens, file_path)
            return tokens

    @staticmethod
    def save_tokens(tokens, file_path):
        """Save tokenized data to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(tokens, f)

    @staticmethod
    def load_tokens(file_path):
        """Load tokenized data from a file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)