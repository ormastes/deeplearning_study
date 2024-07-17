import torch
import torch.nn as nn
import torch.nn.functional as F


# https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class FakeQuantization(nn.Module):
    def __init__(self, scale=1.0, zero_point=0.0):
        super(FakeQuantization, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.zero_point = nn.Parameter(torch.tensor(zero_point), requires_grad=True)

    def forward(self, x):
        q_x = (x / self.scale + self.zero_point).round().to(torch.int32)
        return q_x, self.scale

    # STE https://discuss.pytorch.org/t/the-parameters-of-the-model-with-custom-loss-function-doesnt-upgraded-thorough-its-learning-over-epochs/149024
    def dequantize(self, q_x):
        return (q_x - self.zero_point) * self.scale

    @staticmethod
    def straight_through_estimator(x):
        # Straight-through estimator for quantization
        return x + (x - torch.where(x > 0.5, 1.0, 0.0)).detach()
