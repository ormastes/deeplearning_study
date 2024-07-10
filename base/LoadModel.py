import torch
import numpy as np
def assign_check(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())


def load_weights(gpt, gpt_hf, config):
    d = gpt_hf.state_dict()

    gpt.pos_emb.weight = assign_check(gpt.pos_emb.weight, d["wpe.weight"])
    gpt.tok_emb.weight = assign_check(gpt.tok_emb.weight, d["wte.weight"])

    for b in range(config.num_layers):
        q_w, k_w, v_w = np.split(d[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_query.weight = assign_check(gpt.trf_blocks[b].attn.W_query.weight, q_w.T)
        gpt.trf_blocks[b].attn.W_key.weight = assign_check(gpt.trf_blocks[b].attn.W_key.weight, k_w.T)
        gpt.trf_blocks[b].attn.W_value.weight = assign_check(gpt.trf_blocks[b].attn.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.trf_blocks[b].attn.W_query.bias = assign_check(gpt.trf_blocks[b].attn.W_query.bias, q_b)
        gpt.trf_blocks[b].attn.W_key.bias = assign_check(gpt.trf_blocks[b].attn.W_key.bias, k_b)
        gpt.trf_blocks[b].attn.W_value.bias = assign_check(gpt.trf_blocks[b].attn.W_value.bias, v_b)

        gpt.trf_blocks[b].attn.proj.weight = assign_check(gpt.trf_blocks[b].attn.proj.weight,
                                                             d[f"h.{b}.attn.c_proj.weight"].T)
        gpt.trf_blocks[b].attn.proj.bias = assign_check(gpt.trf_blocks[b].attn.proj.bias,
                                                           d[f"h.{b}.attn.c_proj.bias"])

        gpt.trf_blocks[b].mlp.layers[0].weight = assign_check(gpt.trf_blocks[b].mlp.layers[0].weight,
                                                             d[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].mlp.layers[0].bias = assign_check(gpt.trf_blocks[b].mlp.layers[0].bias,
                                                           d[f"h.{b}.mlp.c_fc.bias"])
        gpt.trf_blocks[b].mlp.layers[3].weight = assign_check(gpt.trf_blocks[b].mlp.layers[3].weight,
                                                             d[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].mlp.layers[3].bias = assign_check(gpt.trf_blocks[b].mlp.layers[3].bias,
                                                           d[f"h.{b}.mlp.c_proj.bias"])

        gpt.trf_blocks[b].norm1.scale = assign_check(gpt.trf_blocks[b].norm1.scale, d[f"h.{b}.ln_1.weight"])
        gpt.trf_blocks[b].norm1.shift = assign_check(gpt.trf_blocks[b].norm1.shift, d[f"h.{b}.ln_1.bias"])
        gpt.trf_blocks[b].norm2.scale = assign_check(gpt.trf_blocks[b].norm2.scale, d[f"h.{b}.ln_2.weight"])
        gpt.trf_blocks[b].norm2.shift = assign_check(gpt.trf_blocks[b].norm2.shift, d[f"h.{b}.ln_2.bias"])

        gpt.final_norm.scale = assign_check(gpt.final_norm.scale, d[f"ln_f.weight"])
        gpt.final_norm.shift = assign_check(gpt.final_norm.shift, d[f"ln_f.bias"])
        gpt.out_head.weight = assign_check(gpt.out_head.weight, d["wte.weight"])


