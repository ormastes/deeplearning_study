import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.input_ids = []
        self.target_ids = []
        self.token_ids = tokenizer.encode(txt)

        for i in range(0, len(self.token_ids) - max_length, stride):
            self.input_ids.append(self.token_ids[i:i + max_length])
            self.target_ids.append(self.token_ids[i + 1:i + 1 + max_length])
            # array[start:stop] stop is exclusive. It is correct to use i+1 here.

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.target_ids[idx])


def create_data_loader(txt, tokenizer, max_length=256, stride=128, batch_size=4, shuffle=True):
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    from BPETokenizer import GPT2TikTokenizer
    tokenizer = GPT2TikTokenizer()
    with open("the-verdict.txt", "r", encoding="utf-8") as file:
        text = file.read()
        data_loader = create_data_loader(text, tokenizer, max_length=4, stride=1, shuffle=False)
        for i, (input_ids, target_ids) in enumerate(data_loader):
            print("Batch:", i)
            print("Input:", input_ids)
            print("Target:", target_ids)
            if i > 2:
                break
