import re

SPLIT_REGEX = re.compile(r'([,.?_!"()\']|--|\s)')
SUB_REGEX = re.compile(r"\s+([,.?_!\"()\'])")


class SimpleTokenizerV1:
    def __init__(self, vocab):
        print("Tokenizer initialized with vocab size:", len(vocab))
        self.str_2_int = vocab
        self.int_2_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocessed_text = SPLIT_REGEX.split(text)
        preprocessed_text = [x.strip() for x in preprocessed_text if x.strip() != ""]
        return [self.str_2_int[x] for x in preprocessed_text if x in self.str_2_int]

    def decode(self, tokens):
        text = " ".join([self.int_2_str[x] for x in tokens])
        text = SUB_REGEX.sub(text)
        return text


if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as file:
        text = file.read()
        preprocessed_text = SPLIT_REGEX.split(text)
        preprocessed_text = [x.strip() for x in preprocessed_text if x.strip() != ""]
        all_tokens = ["<UNK>", "<PAD>", "<EOS>", "<GO>", "<CLS>", "<SEP>", "<MASK>", "<MASK_2>"]
        all_tokens.extend(list(set(preprocessed_text)))
        vocab = {x: i for i, x in enumerate(all_tokens)}
        tokenizer = SimpleTokenizerV1(vocab)
