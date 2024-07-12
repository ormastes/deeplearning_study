import torch
from torch import nn

from base.gpt.BPETokenizer import GPT2TikTokenizer
from base.gpt.GPT2 import GPT2Model
from base.config.GPTConfig import GPT2_CONFIG_124M
from base.util.Util import generate_text_simple


def train(model, n_epochs, save_freq, dataloader, device, config):
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters())
    # Iteration counter
    iter_count = 0
    # Training loop
    for epoch in range(n_epochs):
        model.train()  # Set the model to training mode
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch)
            loss = criterion(outputs.view(-1, config.vocab_size), batch.view(-1))

            loss.backward()
            optimizer.step()

            # Save the model every `save_freq` iterations
            if iter_count % save_freq == 0:
                torch.save(model.state_dict(), f'model_weights_{iter_count}.pth')

            # Increment the iteration counter
            iter_count += 1

        # Evaluation
        model.eval()  # Set the model to evaluation mode
        total_loss = 0
        total_correct = 0
        with torch.no_grad():  # No need to track the gradients
            for batch in dataloader:
                # Move the batch tensors to the right device
                batch = batch.to(device)

                # Forward pass and compute the loss and the number of correct predictions
                outputs = model(batch)
                loss = criterion(outputs.view(-1, config.vocab_size), batch.view(-1))
                total_loss += loss.item()
                total_correct += (outputs.argmax(dim=-1) == batch).sum().item()

        # Print the average loss and accuracy over the epoch
        print(f'Epoch {epoch + 1}/{n_epochs}.. Train loss: {total_loss / len(dataloader):.3f}.. '
              f'Train accuracy: {total_correct / len(dataloader.dataset):.3f}')


def _inference(config, model, device):
    print("Model summary:")
    # data type int32
    #summary(model, (config.ctx_len, config.vocab_size), device="cpu")

    start_context = "Hello, I am"

    tokenizer = GPT2TikTokenizer()
    encoded = tokenizer.encode(start_context)
    # make shape (1, ctx_len) for encoded_tensor and fill 0 for rest of the tokens
    encoded = encoded + [0] * (config.context_length - len(encoded))
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    print("Encoded shape:", encoded_tensor.shape)
    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_len=config.context_length,
    )
    print("Output shape:", out.shape)
    print("Output:", out)
    print("Output text:", tokenizer.decode(out.squeeze(0).tolist()))

def inference(config, device):

    model = GPT2Model(config).to(device)
    _inference(config, model, device )

if __name__ == "__main__":
    config = GPT2_CONFIG_124M()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    inference(config, device)
    exit(0)
    ##############################################################
    # Training GPT2

    # Number of training epochs
    n_epochs = 10

    # Save frequency (in iterations)
    save_freq = 1000

    batch_size = 16

    model_temp_path = "/workspace/data/temp/gpt2/"

    # Load the tokenizer
    tokenizer = GPT2TikTokenizer()

    # Load the dataset
    dataset = load_wikipedia_dataset()


    # Tokenize the dataset
    def tokenize(batch):
        texts = batch['text']
        return [tokenizer.encode(texts[0])]#[tokenizer.encode(text) for text in texts]


    dataset = dataset.map(tokenize, batched=True, batch_size=None)

    # Format the dataset to PyTorch tensors
    dataset.set_format('torch', columns=['input_ids'])

    # Create PyTorch DataLoader
    dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)

    model = GPT2Model(config).to(device)

    # find latest file in model_temp_path dir. file stored time
    import os
    files = os.listdir(model_temp_path)
    file_time_name_map = {}
    files = [f for f in files if f.endswith(".pth")]
    if len(files) == 0:
        print("No model found in", model_temp_path)
    else:
        for file in files:
            file_time_name_map[os.path.getmtime(model_temp_path + file)] = file
        latest_file = file_time_name_map[max(file_time_name_map.keys())]
        print("Loading model:", latest_file)
        model.load_state_dict(torch.load(model_temp_path + latest_file))

    train(model, n_epochs, save_freq, dataloader, device, config)

    ##############################################################
    # Inference

    inference()