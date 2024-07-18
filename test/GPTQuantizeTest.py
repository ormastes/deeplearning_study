import unittest
import os

from base.quantization.QuantizedAttention import QuantizedAttention

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tiktoken
import base.gpt.GPT2 as GPT2
from base.util.Util import *
from transformers import GPT2Model
from base.util.LoadModel import load_weights
from base.config.Config import *
from base.gpt.BPETokenizer import GPT2TikTokenizer
from test.TestUtil import TestUtil

class GPTQuantizeTest(unittest.TestCase):
    def test_quantize_train(self):
        gpt_hf = GPT2Model.from_pretrained(ModelName.gpt2_small_124M.value)
        gpt_hf.eval()

        config = GPT2_CONFIG_124M_TRAIN()
        config.qkv_bias = True
        config.attention = QuantizedAttention
        gpt = GPT2.GPT2Model(config)
        config.context_len = 256  # because data set is too small

        load_weights(gpt, gpt_hf, config)
        torch.manual_seed(123)

        tokenizer = GPT2TikTokenizer()  # tiktoken.get_encoding("gpt2")

        setting = OTHER_SETTINGS(num_epochs=10)

        train_losses, val_losses, tokens_seen, model = train(gpt, config, setting, tokenizer)

        self.assertTrue(train_losses[-1] < 22)
        self.assertTrue(val_losses[-1] < 22)

    def test_quantization_aware_training_train_save_load(self):
        gpt_hf = GPT2Model.from_pretrained(ModelName.gpt2_small_124M.value)
        gpt_hf.eval()

        config = GPT2_CONFIG_124M_TRAIN()
        config.qkv_bias = True
        config.attention = QuantizedAttention
        gpt = GPT2.GPT2Model(config)
        org_context_len = config.context_len
        config.context_len = 256  # because data set is too small

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpt.to(device)
        load_weights(gpt, gpt_hf, config)
        torch.manual_seed(123)

        tokenizer = GPT2TikTokenizer()  # tiktoken.get_encoding("gpt2")

        setting = OTHER_SETTINGS(num_epochs=10)

        train_losses, val_losses, tokens_seen, model = train(gpt, config, setting, tokenizer)

        self.assertTrue(train_losses[-1] < 22)
        self.assertTrue(val_losses[-1] < 22)
        config.no_fake_quantize = True
        config.context_len = org_context_len
        model = TestUtil.logging_after_train(config, model, setting, tokens_seen, train_losses, val_losses)
        model.to(device)
        setting = OTHER_SETTINGS(num_epochs=1)
        config.context_len = 256  # because data set is too small
        train_losses, val_losses, tokens_seen, model = train(model, config, setting, tokenizer, no_train=True)
        self.assertTrue(train_losses[-1] < 22)
        self.assertTrue(val_losses[-1] < 22)

    def test_qlora_train_save_load(self):
        gpt_hf = GPT2Model.from_pretrained(ModelName.gpt2_small_124M.value)
        gpt_hf.eval()

        config = GPT2_CONFIG_124M_TRAIN()
        config.qkv_bias = True
        config.attention = QuantizedAttention
        config.qlora_rank = 4
        gpt = GPT2.GPT2Model(config)
        org_context_len = config.context_len
        config.context_len = 256  # because data set is too small

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpt.to(device)
        load_weights(gpt, gpt_hf, config)
        torch.manual_seed(123)

        tokenizer = GPT2TikTokenizer()  # tiktoken.get_encoding("gpt2")

        setting = OTHER_SETTINGS(num_epochs=10)

        train_losses, val_losses, tokens_seen, model = train(gpt, config, setting, tokenizer)

        self.assertTrue(train_losses[-1] < 22)
        self.assertTrue(val_losses[-1] < 22)
        config.no_fake_quantize = True
        config.context_len = org_context_len
        model = TestUtil.logging_after_train(config, model, setting, tokens_seen, train_losses, val_losses)
        model.to(device)
        setting = OTHER_SETTINGS(num_epochs=1)
        config.context_len = 256  # because data set is too small
        train_losses, val_losses, tokens_seen, model = train(model, config, setting, tokenizer, no_train=True)
        self.assertTrue(train_losses[-1] < 22)
        self.assertTrue(val_losses[-1] < 22)