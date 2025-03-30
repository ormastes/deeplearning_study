import os
import pickle
import torch
import urllib.request
from multiprocessing import Pool, cpu_count
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging
from datasets import load_dataset
from torch.utils.data import DataLoader

from reasoning import ReasoningChat
from reasoning.Config import SimpleConfig
from reasoning.ExactSampleDataset import ExactSampleDataset
from reasoning.ExactSampleTraining import ExactSampleConfig
from reasoning.ReasoningTrainLoop import train_reason
from reasoning.ReasoningTrainLoss import compute_reasoning_penalty

# Suppress warnings from Transformers
hf_logging.set_verbosity_error()


# --------------------------
# Configuration and Settings
# --------------------------
class ReasoningConfig(SimpleConfig):
    def __init__(self):
        super().__init__("exact_trained_model")
        self.learning_rate = 1e-6
        self.weight_decay = 0.1
        self.num_epochs = 1200
        self.max_new_tokens = 1200
        self.temperature = 0.1,
        current_src_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.context_len = 1024  # Maximum sequence length
        self.num_batches = 2  # Batch size for DataLoader
        self.exact_example_config = ExactSampleConfig()
        if os.name == 'nt':
            self.dataset_file = os.path.join(current_src_file_dir, "/manual_data_set/exact_manual_dataset.json")
        else:
            self.dataset_file = "/workspace/data/model/reasoning/raw/cpp_ut_bench_json/train.json"



    def load(self):
        if len(os.listdir(self.model_path)) != 0:
            return super().load()
        else:
            return self.exact_example_config.load()


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
config = ReasoningConfig()

os.makedirs(config.model_path, exist_ok=True)
os.makedirs(config.cache_dir, exist_ok=True)

# --------------------------
# Load Model and Tokenizer
# --------------------------
model, optimizer, tokenizer = config.load()
max_id = tokenizer.vocab_size
torch.autograd.set_detect_anomaly(True)

# --------------------------
# TensorBoard SummaryWriter
# --------------------------
tb_log_dir = os.path.join(config.model_path, "tensorboard_logs")
writer = SummaryWriter(log_dir=tb_log_dir)

dataset = ExactSampleDataset(tokenizer, config)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# --------------------------
# Training Loop with Reasoning Verification
# --------------------------
train_reason(model, optimizer, tokenizer, config, writer, ReasoningChat.chat, train_loader, compute_reasoning_penalty)
# Close the TensorBoard writer after training.
writer.close()
