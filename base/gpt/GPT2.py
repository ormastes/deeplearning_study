import os

from base.gpt.SimpleGPT2Embedding import SimpleGPT2Embedding
from base.gpt.TransformerBlockSequence import SimpleTransformerBlockSequence, SharedTransformerBlockSequence

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from torch import nn

from base.gpt.BPETokenizer import GPT2TikTokenizer
from base.config.GPTConfig import GPT2_CONFIG_124M
from base.util.Util import *
from base.util.Log import *
from base.prim.LayerNorm import LayerNorm
from base.gpt.TransformerBlock import TransformerBlock



class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = Logger.get_instance()
        self.config = config

        self.embedded = SimpleGPT2Embedding(config.vocab_size, config.embed_dim, config.context_length, config)
        self.drop_emb = nn.Dropout(config.drop_rate)

        if config.prim_mum_layers is not None:
            self.trf_blocks = SharedTransformerBlockSequence(config)
        else:
            self.trf_blocks = SimpleTransformerBlockSequence(config)

        self.final_norm = LayerNorm(config.embed_dim)
        self.out_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward_emb(self, x, global_attention_mask):
        self.log.info("Embedding shape:", x.shape)
        x = self.drop_emb(x)
        x = self.trf_blocks(x, global_attention_mask)
        x = self.final_norm(x)
        return x

    def forward(self, tokens, tokenizer):
        global_attention_mask = None
        if self.config.attention_window > 0:
            global_attention_mask = LongformerSelfAttention.update_global_attention_mask(tokens, tokenizer)
        x = self.embedded(tokens)  # Shape [batch_size, num_tokens, emb_size]
        x = self.forward_emb(x, global_attention_mask)
        logits = self.out_head(x)
        return logits

    # device attribute to return self.h device
    @property
    def device(self):
        return next(self.parameters()).device


if __name__ == "__main__":
    Logger.get_instance().level = LogLevel.ERROR
    config = GPT2_CONFIG_124M()
    tokenizer = GPT2TikTokenizer()
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch =torch.stack(batch, dim=0)
    print("Batch shape:", batch.shape)

    torch.manual_seed(123)
    model = GPT2Model(config)
    logits = model(batch)
    print("Logits shape:", logits.shape)
    print("Logits:", logits)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total params:", total_params)

    ##############################################################
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    transformer_block = TransformerBlock(config)
    output = transformer_block(x)
    print("Output shape:", output.shape)

    # for name, param in model.named_parameters():
    #    print(name, "\t", param.numel(),"\t", param.shape)
    #    #print(param)

    ##############################################################
    # Training
    device = torch.device("cuda")
    inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves",
                           [40, 1107, 588]]).to(device)  # "I really like"]

    targets = torch.tensor([[3626, 6100, 345],  # [" effort moves you",
                            [1107, 588, 11311]]).to(device)  # " really like chocolate"]

    tokenizer = GPT2TikTokenizer()
    model.to(device)
    model.train()
    paramters = model.parameters()
    print("Model: GPT2Model", model)
    # print parameters shape
    print("Parameters shape:")
    for p in paramters:
        # print shape with name and shape
        print("Name:", p, "Shape:", p.shape)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    for epoch in range(10):
        for i in range(50):
            optimizer.zero_grad()
            logits = model(inputs)

            logits_flat = logits.flatten(0, 1)
            targets_flat = targets.flatten()
            loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
            perplexity = torch.exp(loss)
            if i == 0:
                #print("Flattened logits:", logits_flat.shape)
                #print("Flattened targets:", targets_flat.shape)
                logit_text = torch.argmax(logits, dim=-1)
                logit_tokens = token_ids_to_text(logit_text[0], tokenizer)
                target_tokens = token_ids_to_text(targets[0], tokenizer)
                print("Output text:", logit_tokens)
                print("Target text:", target_tokens)
                print(loss)
                print(perplexity)
            loss.backward()  # Calculate loss gradients
            optimizer.step()



    ##############################################################
    # Inference
    model = model.to("cpu")

    start_context = "every effort moves"
    tokenizer = GPT2TikTokenizer()
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("Encoded shape:", encoded_tensor.shape)
    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_len=config.context_length
    )
    print("Output shape:", out.shape)
    print("Output:", out)
    print("Output text:", tokenizer.decode(out.squeeze(0).tolist()))

    ##############################################################
    # Original Tokenizer compare
    start_content = "every effort moves"
    tokenizer = GPT2TikTokenizer()
    token_ids = generate_text_simple(model,
                                     text_to_token_ids(start_content, tokenizer, 10),
                                     10, context_len=config.context_length)

    import tiktoken
    orig_tokenizer = tiktoken.get_encoding("gpt2")
    orig_token_ids = generate_text_simple(model,
                                          text_to_token_ids(start_content, tokenizer, 10),
                                          10, context_len=config.context_length)

    print("Token IDs:", token_ids)
    print("Token IDs to Text:", token_ids_to_text(token_ids, tokenizer))

    print("Original Token IDs:", orig_token_ids)
    print("Original Token IDs to Text:", token_ids_to_text(orig_token_ids, orig_tokenizer))
    # compare token_ids
    assert(token_ids.shape == orig_token_ids.shape)
    for (t1, t2) in zip(token_ids, orig_token_ids):
        result = torch.all(t1 == t2)
        assert(result)

    assert(token_ids_to_text(token_ids, tokenizer) == token_ids_to_text(orig_token_ids, orig_tokenizer))

    inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves",
                           [40, 1107, 588]])  # "I really like"]

    targets = torch.tensor([[3626, 6100, 345],  # [" effort moves you",
                            [1107, 588, 11311]])  # " really like chocolate"]

    with torch.no_grad():
        logits = model(inputs)
    # token by softmax
    if False:
        probas = torch.softmax(logits, dim=-1)  # Probability of each token in vocabulary
        print(probas.shape)  # Shape: (batch_size, num_tokens, vocab_size)
        token_ids = torch.argmax(probas, dim=-1, keepdim=True)
        print("Token IDs:\n", token_ids)
        print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
        print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
    # cross entropy in manual
    if False:
        # retrieve the probability of the target token's probas in result
        probas = torch.softmax(logits, dim=-1)  # Probability of each token in vocabulary
        text_idx = 0
        probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
        print("Text 1:", probas_1)

        text_idx = 1
        probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
        print("Text 2:", probas_2)

        # Compute logarithm of all token probabilities
        log_probas = torch.log(torch.cat((probas_1, probas_2)))
        print(log_probas)

        # Calculate the average probability for each token
        avg_log_probas  = torch.mean(log_probas)
        print(avg_log_probas )

        neg_avg_log_probas  = avg_log_probas  * -1
        print(neg_avg_log_probas ) # same as loss of cross entropy
    if True:
        logits_flat = logits.flatten(0, 1)
        targets_flat = targets.flatten()

        print("Flattened logits:", logits_flat.shape)
        print("Flattened targets:", targets_flat.shape)

        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
        print(loss)

        perplexity = torch.exp(loss)
        print(perplexity) # perplexity is more interpretable than cross entropy
        # random similar to token size
        # lower is better prediction


