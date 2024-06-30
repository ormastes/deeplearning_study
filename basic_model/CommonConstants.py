import re

class CommonConstants:
    SPLIT_REGEX = re.compile(r'([,.?_!"()\']|--|\s)')
    SUB_REGEX = re.compile(r"\s+([,.?_!\"()\'])")

    END_OF_TEXT = "<EOS>"
    UNKNOWN_TOKEN = "<UNK>"
    SPECIAL_TOKENS = [UNKNOWN_TOKEN, "<PAD>", "<BOS>", END_OF_TEXT, "<GO>", "<CLS>", "<SEP>", "<MASK>", "<MASK_2>"]
    SPECIAL_TOKENS_SET = set(SPECIAL_TOKENS)
