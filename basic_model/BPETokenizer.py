import importlib
import tiktoken
from CommonConstants import CommonConstants


# Byte Pair Encoding (BPE) Tokenizer
# Byte Pair Encoding (BPE) is a simple form of subword tokenization. It is based on the frequency of the subword units
# It does not require a unknown token, as it can split any word into subword units.
# here we use the tokenizer from tiktoken

class GPT2TikTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        # print the vocab size
        print("Vocab size:", self.tokenizer.n_vocab)

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special=CommonConstants.SPECIAL_TOKENS_SET)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


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
