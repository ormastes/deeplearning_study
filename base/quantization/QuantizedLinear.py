import torch
import torch.nn as nn
import math
from base.quantization.FakeQuantization import FakeQuantization


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, no_fake_quantize=False, num_bits=8):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.no_fake_quantize = no_fake_quantize

        # Initialize weights and biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        self.bias_enabled = bias
        if self.bias_enabled:
            self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=False)

        self.reset_parameters()

        # Initialize learnable quantization parameters for weights, input, and biases
        self.fake_quant_w = FakeQuantization()
        self.fake_quant_x = FakeQuantization()
        self.fake_quant_b = FakeQuantization()

    def reset_parameters(self):
        # original linear use kaiming_uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_enabled:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def update_weights_for_inference(self):
        q_weight, _ = self.fake_quant_w(self.weight)
        self.weight.data = self.fake_quant_w.dequantize(q_weight).data

        if self.bias_enabled:
            q_bias, _ = self.fake_quant_b(self.bias)
            self.bias.data = self.fake_quant_b.dequantize(q_bias).data
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        self.update_weights_for_inference()
        if self.bias_enabled:
            return {
                prefix+'weight': self.weight,
                prefix+'bias': self.bias
            }
        else:
            return {
                 prefix+'weight': self.weight,
            }


    def load_state_dict(self, state_dict, strict=True):
        self.weight.data = state_dict['weight']
        if 'bias' in state_dict:
            assert self.bias_enabled, "Bias is not enabled in the model"
        if self.bias_enabled:
            self.bias.data = state_dict['bias']

    def forward(self, x):
        if self.no_fake_quantize:
            if self.bias_enabled:
                return x @ self.weight.T + self.bias
            else:
                return x @ self.weight.T

        # Quantize weights, input, and biases
        q_weight, scale_w = self.fake_quant_w(self.weight)
        q_input, scale_x = self.fake_quant_x(x)
        if self.bias_enabled:
            q_bias, scale_b = self.fake_quant_b(self.bias)

        # Perform the quantized linear operation
        if self.bias_enabled:
            q_output = q_input @ q_weight.T + q_bias
        else:
            q_output = q_input @ q_weight.T

        # Dequantize the output
        scale_output = scale_w * scale_x
        output = self.fake_quant_w.dequantize(q_output) * scale_output

        return output