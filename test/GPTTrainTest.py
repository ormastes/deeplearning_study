import unittest
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Import from local files
from base.config.Config import GPT2_CONFIG_124M_TRAIN, OTHER_SETTINGS
from base.dataset.SimpleDataset import *
from base.gpt.BPETokenizer import GPT2TikTokenizer
from base.gpt.GPT2 import GPT2Model
from base.config.GPTConfig import GPT2_CONFIG_124M
from base.util.Util import *
from base.util.Log import *


class GPT2Train(unittest.TestCase):
    def test_train(self):
        ###########################
        # Initiate training
        ###########################

        train_losses, val_losses, tokens_seen, model = train(GPT2_CONFIG_124M_TRAIN(), OTHER_SETTINGS())

        self.assertTrue(train_losses[-1] < 2)
        self.assertTrue(val_losses[-1] < 16)

        # compare last train loss with
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

    def test_activity_train(self):
        Logger.get_instance().level = LogLevel.ERROR
        torch.manual_seed(123)  # For reproducibility due to the shuffling in the data loader

        config = GPT2_CONFIG_124M()
        model = GPT2Model(config)
        tokenizer = GPT2TikTokenizer()

        device = torch.device("cuda")
        inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves",
                               [40, 1107, 588]]).to(device)  # "I really like"]

        targets = torch.tensor([[3626, 6100, 345],  # [" effort moves you",
                                [1107, 588, 11311]]).to(device)  # " really like chocolate"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes

        model.train()
        paramters = model.parameters()
        print("Model: GPT2Model", model)
        # print parameters shape
        print("Parameters shape:")
        for p in paramters:
            # print shape with name and shape
            print("Name:", p, "Shape:", p.shape)
        if True:
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
            for epoch in range(6):
                for i in range(50):
                    optimizer.zero_grad()
                    logits = model(inputs)

                    logits_flat = logits.flatten(0, 1)
                    targets_flat = targets.flatten()
                    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
                    perplexity = torch.exp(loss)
                    if i == 0:
                        # print("Flattened logits:", logits_flat.shape)
                        # print("Flattened targets:", targets_flat.shape)
                        logit_text = torch.argmax(logits, dim=-1)
                        logit_tokens = token_ids_to_text(logit_text[0], tokenizer)
                        target_tokens = token_ids_to_text(targets[0], tokenizer)
                        print("Output text:", logit_tokens)
                        print("Target text:", target_tokens)
                        print(loss)
                        print(perplexity)
                    loss.backward()  # Calculate loss gradients
                    optimizer.step()
            # store model and optimizer
            torch.save(model.state_dict(), "model.pth")
            torch.save(optimizer.state_dict(), "optimizer.pth")
        else:
            # load model and optimizer
            model.load_state_dict(torch.load("model.pth"))
            optimizer.load_state_dict(torch.load("optimizer.pth"))

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

        # First 100 characters
        print(text_data[:99])

        total_characters = len(text_data)
        total_tokens = len(tokenizer.encode(text_data))

        print("Characters:", total_characters)
        print("Tokens:", total_tokens)

        train_ratio = 0.9
        train_size = 1 - int(total_tokens * train_ratio)
        train_text = text_data[:train_size]
        val_text = text_data[train_size:]
        torch.manual_seed(123)


        train_loader = create_dataloader_with_worker(train_text, tokenizer,
                                                     batch_size=20,
                                                     max_length=config.context_length,
                                                     stride=config.context_length,
                                                     drop_last=True,
                                                     shuffle=True,
                                                     num_workers=0
                                                     )
        val_loader = create_dataloader_with_worker(val_text, tokenizer,
                                                   batch_size=2,
                                                   max_length=config.context_length,
                                                   stride=config.context_length,
                                                   drop_last=True,
                                                   shuffle=True,
                                                   num_workers=0
                                                   )

        if True:
            print("Train loader:")
            for x, y in train_loader:
                print(x.shape, y.shape)

            print("\nValidation loader:")
            for x, y in val_loader:
                print(x.shape, y.shape)

            with torch.no_grad():  # Disable gradient tracking for efficiency because we are not training, yet
                train_loss = calc_loss_loader(train_loader, model)
                val_loss = calc_loss_loader(val_loader, model)

            print("Training loss:", train_loss)
            print("Validation loss:", val_loss)

        if True:
            # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

            num_epochs = 1000

            Logger.get_instance().level = LogLevel.ERROR
            train_losses, val_losses, tokens_seen = train_model_simple(
                model, train_loader, val_loader, optimizer,
                num_epochs=num_epochs, eval_freq=5, eval_iter=5,
                start_context="Every effort moves", tokenizer=tokenizer
            )


if __name__ == '__main__':
    unittest.main()
