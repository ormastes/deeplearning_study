from ctransformers import AutoModelForCausalLM
import torch
import os
# --------------------------
# Configuration and Settings
# --------------------------
class SimpleConfig:
    def __init__(self, model_name):
        self.learning_rate = 1e-6
        self.weight_decay = 0.1
        self.num_epochs = 1200
        self.max_new_tokens = 12000
        self.temperature = 0.1,
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        current_src_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_src_file_dir, self.model_name,
                                       "saved_models")  # Directory to save models and checkpoints
        self.cache_dir = os.path.join(current_src_file_dir, self.model_name,
                                      "cache")  # Cache directory for tokenized data
        self.context_len = 256  # Maximum sequence length
        self.num_batches = 2  # Batch size for DataLoader
        if os.name == 'nt':
            self.original_model_path = os.path.join(current_src_file_dir, "/raw/WizardCoderModel")
            self.dataset_file = os.path.join(current_src_file_dir, "/manual_data_set/exact_manual_dataset.json")
        else:
            self.original_model_path = "/workspace/data/model/reasoning/raw/WizardCoderModel"
            self.dataset_file = "/workspace/data/model/reasoning/raw/cpp_ut_bench_json/train.json"

    def save(self, epoch, model, optimizer):
        # Save checkpoint including optimizer state.
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(self.model_path, f"{self.model_name}_model_{epoch}.pt"))

    def _load(self, model_save_dir):
        files = os.listdir(model_save_dir)
        model = AutoModelForCausalLM.from_pretrained(self.original_model_path)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if len(files) > 0:
            # order by epoch
            files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            # get the last file
            model_file = os.path.join(model_save_dir, files[-1])
            checkpoint = torch.load(model_file, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.to(self.device)
        for state in optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(self.device)
            elif isinstance(state, dict):
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.device)
        return model, optimizer

    def load(self):
        # Load latest checkpoint.
        # get all file in the model path
        return self._load(self.model_path)


    def log_answer(self, error_text, answer, idx):
        print(error_text, answer, idx)