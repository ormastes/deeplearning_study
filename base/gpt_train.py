# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import matplotlib.pyplot as plt
import os
import torch
import urllib.request

# Import from local files
from base.Util import *
from base.Config import GPT2_CONFIG_124M_TRAIN, OTHER_SETTINGS
from base.GPT2 import GPT2Model
from base.SimpleDataset import create_dataloader_with_worker


def main(gpt_config, settings):
    from base.Log import Logger, LogLevel
    Logger.get_instance().level = LogLevel.ERROR
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # Download data if necessary
    ##############################

    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    ##############################
    # Initialize model
    ##############################


    model = GPT2Model(gpt_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    from BPETokenizer import GPT2TikTokenizer
    tokenizer = GPT2TikTokenizer() #tiktoken.get_encoding("gpt2")

    train_loader = create_dataloader_with_worker(
        text_data[:split_idx], tokenizer,
        batch_size=settings.batch_size,
        max_length=gpt_config.context_length,
        stride=gpt_config.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_with_worker(
        text_data[split_idx:], tokenizer,
        batch_size=settings.batch_size,
        max_length=gpt_config.context_length,
        stride=gpt_config.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    ##############################
    # Train model
    ##############################


    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer,
        num_epochs=settings.num_epochs, eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":



    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(GPT2_CONFIG_124M_TRAIN(), OTHER_SETTINGS())

    ###########################
    # After training
    ###########################

    # Plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS().num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), "model.pth")
    model = GPT2Model(GPT2_CONFIG_124M_TRAIN())
    model.load_state_dict(torch.load("model.pth"))