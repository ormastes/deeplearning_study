import torch
from torch.utils.data import Dataset
import pickle

class ExactSampleDataset(Dataset):
    """
    Custom Dataset to load training samples from a plain text dataset file.
    Each sample is delimited by the marker '####SAMPLE_END####'.
    """
    def __init__(self, tokenizer, config, stride=4, transform=None):
        self.input_ids = []
        self.target_ids = []

        with open(config.dataset_file, "rb") as f:
            token_lists = pickle.load(f)
        token_ids = []
        for token_list in token_lists:
            token_ids.extend( token_list)

        print("# of tokens in txt:", len(token_ids))

        for i in range(0, len(token_ids) - config.max_length, stride):
            input_chunk = token_ids[i:i + config.max_length+1]
            self.input_ids.append(input_chunk)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        try:
            return self.input_ids[idx]
        except Exception as e:
            print(f"Error getting sample {idx}: {e}")
            return None



# Example usage for training:
