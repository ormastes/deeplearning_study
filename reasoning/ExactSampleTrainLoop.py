import torch
import os
from transformers import AdamW

def get_training_layers(model):
    # get first and last layer's parameters
    params = list(model.parameters())
    # merge first and last layer's parameters
    return params[0:2] + params[-2:]

def train_exact(
        model,           # The model being trained
        optimizer,       # Optimizer for training
        tokenizer,       # Tokenizer for encoding/decoding
        config,
        writer,
        chat,                   # Function to generate output
        train_loader,           # Training questions
        reward_fn,              # Function to calculate rewards
        num_epochs,             # Number of training epochs
        group_size=16,          # Number of outputs to sample per question
        learning_rate=1e-6,     # Learning rate
        validation_batches=10,  # Number of batches to use for validation
        model_path=None,        # Path to save model checkpoints
        module_name="reasoning",
        epsilon=0.2,            # PPO clipping parameter
        beta=0.04               # KL penalty coefficient
):
    # Store the current policy as the old policy
    #old_policy = copy.deepcopy(model)
    prams_to_train = get_training_layers(model)
    optimizer = AdamW(prams_to_train, lr=learning_rate)
    model.train()
    
    global_step = 0
    for epoch in range(num_epochs):
        idx = 0
        epoch_loss = 0.0
        for item_idx, question_ids, answer_ids in enumerate(train_loader):
            if idx % group_size == 0:
                optimizer.zero_grad()
            idx += 1

            outputs = model(input_ids=question_ids, labels=answer_ids)
            loss = outputs.loss

            # Update policy model
            loss.backward()
            if idx % group_size == 0:
                optimizer.step()

            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            global_step += 1
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Train/EpochLoss", avg_epoch_loss, epoch)
        print(f"Epoch {epoch} average training loss: {avg_epoch_loss:.4f}")
        
        # --------------------------
        # Validation Phase (with reasoning test)
        # --------------------------
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                for idx, question_ids, answer_ids in enumerate(train_loader):
                    outputs = model(input_ids=question_ids, labels=answer_ids)
                    loss = outputs.loss

                    # Update policy model
                    loss.backward()
                    if idx % group_size == 0:
                        optimizer.step()

                    writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
                    global_step += 1
                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / len(train_loader)
                writer.add_scalar("Train/EpochLoss", avg_epoch_loss, epoch)
                print(f"Epoch {epoch} average training loss: {avg_epoch_loss:.4f}")
            model.train()
            config.save(epoch, model, optimizer)

    return model