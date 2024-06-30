import torch
from basic_model.SimpleUtil import simple_softmax


class SimpleAttention_v1:
    KQV_DIM = 3
    Q = 0
    K = 1
    V = 2

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = torch.nn.Parameter(torch.rand(SimpleAttention_v1.KQV_DIM, input_dim, output_dim, requires_grad=False))

    def __call__(self, inputs):
        queries = inputs @ self.weight[SimpleAttention_v1.Q]
        keys = inputs @ self.weight[SimpleAttention_v1.K]
        values = inputs @ self.weight[SimpleAttention_v1.V]
        print("QKV values:", queries, keys, values)
        attention_scores = queries @ keys.T
        d_k = keys.shape[-1]
        # linear scale down by square root of d_k
        #attention_weights = torch.nn.functional.softmax(attention_scores / d_k ** 0.5, dim=-1)
        attention_weights = simple_softmax(attention_scores / d_k**0.5, dim=attention_scores.ndim-1)
        attended_context = attention_weights @ values
        return attended_context

    def __repr__(self):
        return f"SimpleAttention(input_dim={self.input_dim}, output_dim={self.output_dim})"

if __name__ == "__main__":
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )
    ##############################################################
    # Attention applied to a token embedding
    print("Input shape:", inputs.shape)
    query = inputs[1]
    print("Query:", query)

    attention_scores = torch.empty(inputs.shape[0])
    print("Attention shape:", attention_scores.shape)

    for i, x in enumerate(inputs):
        attention_scores[i] = torch.dot(query, x)
    print("Attention scores:", attention_scores)
    attention_weights = attention_scores/attention_scores.sum()
    print("Attention weights:", attention_weights)
    print("Attention weights Sum:", attention_weights.sum())

    attention_weights_naive = simple_softmax(attention_scores)
    print("Attention weights (naive):", attention_weights_naive)
    print("Attention weights Sum (naive):", attention_weights_naive.sum())

    attention_weights_lib = torch.nn.functional.softmax(attention_scores, dim=0)
    print("Attention weights (lib):", attention_weights_lib)
    print("Attention weights Sum (lib):", attention_weights_lib.sum())

    # transpose attention and repeat columns by input's columns to matmul with inputs
    attention_weights_to_matmul = attention_weights_naive # add 0 size dimension on specified location.
    print("Attention weights to matmul:", attention_weights_to_matmul)
    print("Attention weights to matmul shape:", attention_weights_to_matmul.shape)
    attended_context = attention_weights_to_matmul @ inputs # matmul
    print("Attended context:", attended_context)
    print("Attended context shape:", attended_context.shape)

    ##############################################################
    # Attention applied to all token embeddings
    queries = inputs.T
    attention_scores = inputs @ queries
    print("Attention scores:", attention_scores)
    attention_weights = simple_softmax(attention_scores, dim=1)
    attention_weights2 = torch.nn.functional.softmax(attention_scores, dim=1)
    print("Attention weights:", attention_weights)
    all_context = attention_weights @ inputs
    print("All context:", all_context)

    ##############################################################
    # Using simple attention class
    d_in = inputs.shape[1]
    d_out = 2
    torch.manual_seed(123)
    W = torch.nn.Parameter(torch.rand(3, d_in, d_out, requires_grad=False))

    query = inputs @ W[0]
    key = inputs @ W[1]
    value = inputs @ W[2]
    print("QKV values:", query, key, value)
    attention_score = query @ key.T
    print("Attention score:", attention_score)
    d_k = key.shape[-1]
    attention_weights = simple_softmax(attention_score / d_k ** 0.5, dim=attention_score.ndim - 1)
    print("Attention weights:", attention_weights)
    attended_context = attention_weights @ value
    print("Attended context:", attended_context)

    torch.manual_seed(123)
    simple_attention = SimpleAttention_v1(d_in, d_out)
    print(simple_attention(inputs))
