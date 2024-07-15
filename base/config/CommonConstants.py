import re

class CommonConstants:
    SPLIT_REGEX = re.compile(r'([,.?_!"()\']|--|\s)')
    SUB_REGEX = re.compile(r"\s+([,.?_!\"()\'])")

    UNKNOWN_TOKEN = "<UNK>"
    END_OF_TEXT = "<EOS_SEQ>"  # End of sequence

    END_OF_SENTENCE = "<EOS_SENT>"  # End of sentence
    END_OF_PARAGRAPH = "<EOS_PARA>"  # End of paragraph
    QUESTION_TOKEN = "<QUESTION>"
    ANSWER_TOKEN = "<ANSWER>"
    CLASSIFICATION_TOKEN = "<CLS>"
    SEPARATOR_TOKEN = "<SEP>"

    IMPORTANT_TOKENS = [END_OF_SENTENCE, END_OF_PARAGRAPH, QUESTION_TOKEN, ANSWER_TOKEN, CLASSIFICATION_TOKEN, SEPARATOR_TOKEN]

    SPECIAL_TOKENS = [UNKNOWN_TOKEN, "<PAD>", "<GO>", "<MASK>", "<MASK_2>"]
    SPECIAL_TOKENS.extend(IMPORTANT_TOKENS)
    SPECIAL_TOKENS_SET = set(SPECIAL_TOKENS)

