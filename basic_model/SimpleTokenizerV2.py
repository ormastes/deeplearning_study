from CommonConstants import CommonConstants


class SimpleTokenizerV2:
    def __init__(self, vocab):
        print("Tokenizer initialized with vocab size:", len(vocab))
        self.str_2_int = vocab
        self.int_2_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocessed_text = CommonConstants.SPLIT_REGEX.split(text)
        preprocessed_text = [x.strip() for x in preprocessed_text if x.strip() != ""]
        preprocessed_text = [x if x in self.str_2_int else CommonConstants.UNKNOWN_TOKEN for x in preprocessed_text]
        return [self.str_2_int[x] for x in preprocessed_text if x in self.str_2_int]

    def decode(self, tokens):
        text = " ".join([self.int_2_str[x] for x in tokens])
        text = CommonConstants.SUB_REGEX.sub(r'\1', text)
        return text


if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as file:
        text = file.read()
        preprocessed_text = CommonConstants.SPLIT_REGEX.split(text)
        preprocessed_text = [x.strip() for x in preprocessed_text if x.strip() != ""]
        all_tokens = CommonConstants.SPECIAL_TOKENS
        all_tokens.extend(list(set(preprocessed_text)))
        vocab = {x: i for i, x in enumerate(all_tokens)}
        tokenizer = SimpleTokenizerV2(vocab)
        text1 = "Hello, do you like tea?"
        text2 = "In the sunlit terraces of the place."
        text = (" "+CommonConstants.END_OF_TEXT+" ").join([text1, text2])
        print("Original text:", text)
        tokens = tokenizer.encode(text)
        print("Tokens:", tokens)
        decoded_text = tokenizer.decode(tokens)
        print("Decoded text:", decoded_text)
