import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        #print("GPT2 output shape:", logits.shape)
        #probas = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, idx_next], dim=-1)

        #next_token = torch.argmax(logits, dim=-1)
        #idx = torch.cat([idx, next_token], dim=-1)
    return idx


def text_to_token_ids(text, tokenizer, max_len):
    token_ids = tokenizer.encode(text)
    #token_ids = token_ids + [0] * (max_len - len(token_ids))

    return torch.tensor(token_ids).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(inputs, targets, model):
    inputs = inputs.to(model.device)
    targets = targets.to(model.device)
    logits = model(inputs)
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    return loss


def calc_loss_loader(dataloader, model, num_batches=None):
    total_loss = 0
    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for i, (inputs, targets) in enumerate(dataloader):
        loss = calc_loss_batch(inputs, targets, model)
        total_loss += loss.item()
        if i >= num_batches:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model,  num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer, context_size).to(model.device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            # Optional evaluation step
            if False:#global_step % eval_freq == 0:
                input_1 = input_batch[0]
                target_1 = target_batch[0]
                print("Input text:", token_ids_to_text(input_1, tokenizer))
                print("Target text:", token_ids_to_text(target_1, tokenizer))
                print("")
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                import math
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f} perplexity {math.exp(loss):.3f} ,"
                      f" Val loss {val_loss:.3f} perplexity {math.exp(val_loss):.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, start_context
        )

    return train_losses, val_losses, track_tokens_seen

import matplotlib.pyplot as plt

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()
