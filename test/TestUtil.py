import unittest
import os

from base.gpt.TransformerBlockSequence import SharedTransformerBlockSequence
from base.quantization.QuantizedAttention import QuantizedAttention

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Import from local files
from base.config.Config import GPT2_CONFIG_124M_TRAIN, OTHER_SETTINGS
from base.dataset.SimpleDataset import *
from base.gpt.BPETokenizer import GPT2TikTokenizer
from base.gpt.GPT2 import GPT2Model
from base.config.GPTConfig import GPT2_CONFIG_124M
from base.util.Util import *
from base.util.Log import *
from base.embedding.AttentionLinearBiasPositionalEmbedding import *

class TestUtil:
    @staticmethod
    def compare_parameters(model, new_model):
        new_model.to(model.device)
        new_model_params = new_model.state_dict()
        model_params = model.state_dict()
        TestUtil._compare_parameters_recursive(model_params, new_model_params)

    @staticmethod
    def _compare_parameters_recursive(model_params, new_model_params):
        for name in model_params.keys():
            param = model_params[name]
            if isinstance(param, dict):
                TestUtil._compare_parameters_recursive(param, new_model_params[name])
            elif isinstance(param, torch.Tensor):
                new_model_param = new_model_params[name]
                if not torch.equal(param, new_model_param):
                    print(f"Parameter {name} is not equal")
            else:
                print(f"Parameter {name} is not required grad")

    @staticmethod
    def apply_original_weight(model):
        # loop child modules
        for name, module in model.named_children():
            # check initialize_from_existing_weights function exist
            if hasattr(module, 'initialize_from_existing_weights'):
                module.initialize_from_existing_weights()
            else:
                TestUtil.apply_original_weight(module)

    @staticmethod
    def logging_after_train(config, model, setting, tokens_seen, train_losses, val_losses, file_name="model.pth"):
        # compare last train loss with
        ###########################
        # After training
        ###########################
        # Plot results
        epochs_tensor = torch.linspace(0, setting.num_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        plt.savefig("loss.pdf")
        # Save and load model
        torch.save(model.state_dict(), file_name)
        new_model = GPT2Model(config)
        new_model.load_state_dict(torch.load(file_name))

        return new_model