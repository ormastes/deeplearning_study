import torch
import torch.nn as nn
import math
from base.quantization.FakeQuantization import FakeQuantization


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_bits=8):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits

        # Initialize weights and biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        # Initialize learnable quantization parameters for weights, input, and biases
        self.fake_quant_w = FakeQuantization()
        self.fake_quant_x = FakeQuantization()
        self.fake_quant_b = FakeQuantization()

    def forward(self, x):
        # Quantize weights, input, and biases
        q_weight, scale_w = self.fake_quant_w(self.weight)
        q_input, scale_x = self.fake_quant_x(x)
        q_bias, scale_b = self.fake_quant_b(self.bias)

        # Perform the quantized linear operation
        q_output = torch.matmul(q_weight, q_input) + q_bias

        # Dequantize the output
        scale_output = scale_w * scale_x
        output = self.fake_quant_w.dequantize(q_output) * scale_output

        return output