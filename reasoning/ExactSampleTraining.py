import os
import pickle
import torch
import urllib.request
from multiprocessing import Pool, cpu_count
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging
from torch.utils.data import DataLoader

from reasoning import ReasoningChat
from reasoning.ExactSampleDataset import ExactSampleDataset
from reasoning.ExactSampleTrainLoop import train_exact
from reasoning.Config import SimpleConfig

# Suppress warnings from Transformers
hf_logging.set_verbosity_error()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --------------------------
# Configuration and Settings
# --------------------------
class ExactSampleConfig(SimpleConfig):
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-6
        self.weight_decay = 0.1
        self.num_epochs = 1200
        self.max_new_tokens = 0.1
        self.context_len = 256  # Maximum sequence length
        self.num_batches = 2  # Batch size for DataLoader


# --------------------------
# Logger Stub
# --------------------------
class LogLevel:
    ERROR = 40


class Logger:
    _instance = None

    def __init__(self):
        self.level = LogLevel.ERROR

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Logger()
        return cls._instance


# --------------------------
# Instantiate Config and Settings
# --------------------------
config = ExactSampleConfig()

# --------------------------
# Load Model and Tokenizer
# --------------------------
model, optimizer, tokenizer = config.load()
max_id = tokenizer.vocab_size
torch.autograd.set_detect_anomaly(True)
print("Padding side:", tokenizer.padding_side)

# --------------------------
# TensorBoard SummaryWriter
# --------------------------
tb_log_dir = os.path.join(config.model_path, "tensorboard_logs")
writer = SummaryWriter(log_dir=tb_log_dir)

def collate_fn(batch):
    m = max([batch[i].shape[0] for i in range(len(batch))])
    pad = tokenizer.pad_token_id
    # fill padding tokens
    batch = [torch.cat([batch[i], torch.tensor([pad] * (m - batch[i].shape[0]), dtype=torch.long)]) for i in range(len(batch))]
    batch = torch.stack(batch, dim=0)
    return batch

dataset = ExactSampleDataset(tokenizer, config)
print("Dataset length:", len(dataset))
train_loader = DataLoader(dataset, batch_size=config.num_batches, num_workers=8,collate_fn=collate_fn) #shuffle=True,
print("Train Loader done")

from torchsummary import summary

#summary(model, (2, 256, tokenizer.vocab_size))
for name, param in model.named_parameters():
    print(name, param.size())

# --------------------------
# Training Loop with Reasoning Verification
# --------------------------
train_exact(model, optimizer, tokenizer, config, writer, ReasoningChat.chat, train_loader)
# Close the TensorBoard writer after training.
writer.close()
