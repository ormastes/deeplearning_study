import torch


def simple_softmax(x, dim=0):
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    e_x_sum = e_x.sum(dim=dim, keepdim=True)
    return e_x / e_x_sum