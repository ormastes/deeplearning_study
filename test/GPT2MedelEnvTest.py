import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import unittest
import torch
from transformers import GPT2Tokenizer, GPT2Model

class GPT2ModelEvnTest(unittest.TestCase):
    def test_gpt2_model(self):
        #tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GPT2Model.from_pretrained('gpt2')
        #text = "Replace me by any text you'd like."
        #encoded_input = tokenizer(text, return_tensors='pt')
        #output = model(**encoded_input)
        #summary(model, (1024, 768), dtype=torch.IntTensor)
        total_params = sum(p.numel() for p in model.parameters())
        print("Total params:", total_params)
        for name, param in model.named_parameters():
            print(name, "\t", param.numel(),"\t", param.shape)


    def test_gpt2_model_simple_response(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2').to(device)
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors='pt').input_ids.to(device)
        output = model(encoded_input)
        self.assertTrue(output is not None)


    def test_gpt2_homemade(self):
        from base.gpt.GPT2 import GPT2Model, GPT2_CONFIG_124M
        config = GPT2_CONFIG_124M()
        model = GPT2Model(config)
        total_params = sum(p.numel() for p in model.parameters())
        print("Total params:", total_params)
        for name, param in model.named_parameters():
            print(name, "\t", param.numel(),"\t", param.shape)

if __name__ == '__main__':
    unittest.main()
