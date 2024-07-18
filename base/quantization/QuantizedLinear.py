import torch
import torch.nn as nn
import math
from base.quantization.FakeQuantization import FakeQuantization


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, config, bias=True, qlora_rank=None, no_fake_quantize=None, num_bits=8):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.no_fake_quantize = config.no_fake_quantize if no_fake_quantize is None else no_fake_quantize
        self.qlora_rank = config.qlora_rank if qlora_rank is None else qlora_rank
        self.bias_enabled = config.linear_bias if bias is None else bias
        self.aq_num_codebooks = config.aq_num_codebooks
        self.mcq_num_codebooks = config.mcq_num_codebooks
        self.mcp_codebook_size = config.mcq_codebook_size

        #         self.aq_num_codebooks = 0
        #         self.mcq_num_codebooks = 0
        #         self.mcp_codebook_size = 0
        # Initialize weights and biases
        if self.aq_num_codebooks != 0:
            self.qweight = nn.Parameter(torch.Tensor(self.aq_num_codebooks, in_features, out_features), requires_grad=True)
        elif self.mcq_num_codebooks != 0:
            # Initialize codebooks
            self.qweight = nn.Parameter(torch.Tensor(self.mcp_codebook_size, in_features), requires_grad=True)
            self.codeword_indices = nn.Parameter(torch.randint(0, self.mcp_codebook_size,
                                                               (out_features, self.mcq_num_codebooks)
                                                               ), requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        self.bias_enabled = bias
        if self.bias_enabled:
            self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=False)

        if self.qlora_rank != 0:
            # Low-rank adaptation matrices
            self.qlora_A = nn.Parameter(torch.randn(in_features, self.qlora_rank))
            self.qlora_B = nn.Parameter(torch.randn(self.qlora_rank, out_features))

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

        if self.qweight is not None:
            nn.init.kaiming_uniform_(self.qweight, a=math.sqrt(5))

        if self.qlora_rank != 0:
            nn.init.kaiming_uniform_(self.qlora_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.qlora_B, a=math.sqrt(5))

    def initialize_from_existing_weights(self):
        if self.aq_num_codebooks != 0:
            self.qweight[0].data.copy_(self.weight)
        elif self.mcq_num_codebooks != 0:
            qweight_row = self.qweight.shape[0]
            weight_row = self.weight.shape[0]
            if qweight_row > weight_row:
                self.qweight[:weight_row, :].data.copy_(self.weight)
            elif qweight_row < weight_row:
                self.qweight.data.copy_(self.weight[:qweight_row, :])
            else:
                self.qweight[0].data.copy_(self.weight)


    def update_weights_for_inference(self):
        q_weight, _ = self.fake_quant_w(self.weight)
        self.weight.data = self.fake_quant_w.dequantize(q_weight).data

        if self.bias_enabled:
            q_bias, _ = self.fake_quant_b(self.bias)
            self.bias.data = self.fake_quant_b.dequantize(q_bias).data

        if self.qlora_rank != 0:
            q_qlora_A, _ = self.fake_quant_w(self.qlora_A)
            q_qlora_B, _ = self.fake_quant_w(self.qlora_B)
            self.qlora_A.data = self.fake_quant_w.dequantize(q_qlora_A).data
            self.qlora_B.data = self.fake_quant_w.dequantize(q_qlora_B).data

        if self.aq_num_codebooks != 0:
            q_qweight, _ = self.fake_quant_w(self.qweight)
            self.qweight.data = self.fake_quant_w.dequantize(q_qweight).data
        elif self.mcq_num_codebooks != 0:
            # Initialize codebooks
            q_qweight, _ = self.fake_quant_w(self.qweight)
            self.qweight.data = self.fake_quant_w.dequantize(q_qweight).data


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        self.update_weights_for_inference()
        state = {
            prefix + 'weight': self.weight,
        }
        if self.bias_enabled:
            state[prefix + 'bias'] = self.bias

        if self.qlora_rank != 0:
            # Low-rank adaptation matrices
            state[prefix + 'qlora_A'] = self.qlora_A
            state[prefix + 'qlora_B'] = self.qlora_B

        if self.aq_num_codebooks != 0:
            state[prefix + 'qweight'] = self.qweight

        if self.mcq_num_codebooks != 0:
            state[prefix + 'qweight'] = self.qweight
            state[prefix + 'codeword_indices'] = self.codeword_indices

        return state

    def load_state_dict(self, state_dict, strict=True):
        self.weight.data = state_dict['weight']
        if 'bias' in state_dict:
            assert self.bias_enabled, "Bias is not enabled in the model"
        if self.bias_enabled:
            self.bias.data = state_dict['bias']

        if 'qlora_A' in state_dict:
            assert self.qlora_rank != 0, "Low-rank adaptation matrices are not enabled in the model"
        if self.qlora_rank != 0:
            self.qlora_A.data = state_dict['qlora_A']
            self.qlora_B.data = state_dict['qlora_B']

        if 'qweight' in state_dict:
            assert self.aq_num_codebooks != 0 or self.mcq_num_codebooks != 0, "Adaptive quantization is not enabled in the model"
        if self.aq_num_codebooks != 0:
            self.qweight.data = state_dict['qweight']

        if self.mcq_num_codebooks != 0:
            self.qweight.data = state_dict['qweight']
            self.codeword_indices.data = state_dict['codeword_indices']

    def _forward(self, input, weight, bias):
        # Perform the quantized linear operation
        if self.bias_enabled:
            output = input @ weight.T + bias
        else:
            output = input @ weight.T
        return output

    def forward(self, x):
        weight = self.weight
        additional_weight = 0.0
        weight_quantized = False
        input = x
        if self.bias_enabled:
            bias = self.bias

        if self.qlora_rank != 0:
            qlora_A = self.qlora_A
            qlora_B = self.qlora_B
            if not self.no_fake_quantize:
                qlora_A, _ = self.fake_quant_w(qlora_A)
                qlora_B, _ = self.fake_quant_w(qlora_B)
            additional_weight = qlora_A @ qlora_B

        if self.aq_num_codebooks != 0:
            weight = self.qweight
            if not self.no_fake_quantize:
                weight, scale_w = self.fake_quant_w(weight)
            weight = torch.sum(weight, dim=0)
        elif self.mcq_num_codebooks != 0:
            # Perform matrix multiplication with multiple codebooks
            weight = self.qweight
            if not self.no_fake_quantize:
                weight, scale_w = self.fake_quant_w(weight)

            new_weight = None
            for i in range(self.mcq_num_codebooks):
                codebook = torch.index_select(weight, 0, self.codeword_indices[:, i])
                new_weight = codebook if new_weight is None else new_weight + codebook
            weight = new_weight
        else:
            if not self.no_fake_quantize:
                weight, scale_w = self.fake_quant_w(weight)

        if not self.no_fake_quantize:
            # Quantize weights, input, and biases
            # weight, scale_w = self.fake_quant_w(weight)
            input, scale_x = self.fake_quant_x(input)
            scale_output = scale_w * scale_x
            if self.bias_enabled:
                bias, _ = self.fake_quant_b(bias)

        output = self._forward(input, weight + additional_weight, bias)

        if self.no_fake_quantize:
            return output
        else:
            # Dequantize the output
            return self.fake_quant_w.dequantize(output) * scale_output
