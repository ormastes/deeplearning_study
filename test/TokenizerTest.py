import unittest
import os

from base.gpt import GPT2
from base.gpt.TransformerBlockSequence import SharedTransformerBlockSequence
from base.quantization.QuantizedAttention import QuantizedAttention

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Import from local files
from base.config.Config import GPT2_CONFIG_124M_TRAIN, OTHER_SETTINGS, GPT2_CONFIG_124M_TRAIN_SMALL_CONTEXT
from base.dataset.SimpleDataset import *
from base.gpt.BPETokenizer import GPT2TikTokenizer
from base.gpt.GPT2 import GPT2Model
from base.config.GPTConfig import GPT2_CONFIG_124M
from base.util.Util import *
from base.util.Log import *
from base.embedding.AttentionLinearBiasPositionalEmbedding import *
from test.TestUtil import TestUtil
from base.gpt.FlashAttention import FlashAttention

class TokenizerTest(unittest.TestCase):
    def test_synonym_token(self):
        pass
