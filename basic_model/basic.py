from BPETokenizer import GPT2TikTokenizer


if __name__ == "__main__":
    print("This is the basic model.")
    with open("the-verdict.txt", "r", encoding="utf-8") as file:
        text = file.read()
        print("Number of characters in the file:", len(text))
        tokenizer = GPT2TikTokenizer()
        tokens = tokenizer.encode(text)
        print("Number of tokens:", len(tokens))
        token_sample = tokens[50:] # remove the first 50 tokens
        context_size = 4
        for i in range(1, context_size+1):
            context = token_sample[:i]
            expected = token_sample[i]
            print(context, "--->", expected)

        for i in range(1, context_size+1):
            context = token_sample[:i]
            expected = token_sample[i]
            print(tokenizer.decode(context), "--->", tokenizer.decode([expected]))


