import unittest
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tiktoken
import base.gpt.GPT2 as GPT2
from base.util.Util import *
from transformers import GPT2Model
from base.util.LoadModel import load_weights
from base.config.Config import *
from base.gpt.BPETokenizer import GPT2TikTokenizer


class MyTestCase(unittest.TestCase):
    def test_load(self):
        gpt_hf = GPT2Model.from_pretrained(ModelName.gpt2_small_124M.value)
        gpt_hf.eval()

        BASE_CONFIG = model_configs[ModelName.gpt2_small_124M]

        gpt = GPT2.GPT2Model(BASE_CONFIG)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_weights(gpt, gpt_hf, BASE_CONFIG)
        torch.manual_seed(123)

        tokenizer = GPT2TikTokenizer() # tiktoken.get_encoding("gpt2")

        # def generate_text(model, idx, max_new_tokens, context_size, tokenizer, temperature=0.0, top_k=None, eos_id=CommonConstants.END_OF_TEXT, is_org=False):
        token_ids = generate_text(
            model=gpt.to(device),
            idx=text_to_token_ids("Every effort moves", tokenizer).to(device),
            max_new_tokens=30,
            context_size=BASE_CONFIG.context_len,
            tokenizer=tokenizer,
            top_k=1,
            temperature=1.0
        )
        expected_output = "I'm not going to sit here and say, 'I'm not going to do this,'"
        self.assertTrue(expected_output in token_ids)

    def test_load_validation(self):
        gpt_hf = GPT2Model.from_pretrained(ModelName.gpt2_small_124M.value)
        gpt_hf.eval()

        config = GPT2_CONFIG_124M_TRAIN()
        config.qkv_bias = True
        gpt = GPT2.GPT2Model(config)
        config.context_len = 256  # because data set is too small

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpt.to(device)
        load_weights(gpt, gpt_hf, config)
        torch.manual_seed(123)

        tokenizer = GPT2TikTokenizer() # tiktoken.get_encoding("gpt2")

        setting = OTHER_SETTINGS(num_epochs=1)

        train_losses, val_losses, tokens_seen, model = train(gpt, config, setting, tokenizer)

        self.assertTrue(train_losses[-1] < 22)
        self.assertTrue(val_losses[-1] < 22)


if __name__ == '__main__':
    unittest.main()
