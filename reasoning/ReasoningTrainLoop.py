import torch
import os
from transformers import AdamW

def get_training_layers(model):
    # get first and last layer's parameters
    params = list(model.parameters())
    # merge first and last layer's parameters
    return params[0:2] + params[-2:]

def train_reason(
        model,           # The model being trained
        optimizer,       # Optimizer for training
        tokenizer,       # Tokenizer for encoding/decoding
        config,
        writer,
        chat,                   # Function to generate output
        train_loader,           # Training questions
        reward_fn,              # Function to calculate rewards
        group_size=16,          # Number of outputs to sample per question
        learning_rate=1e-6,     # Learning rate
        validation_batches=10,  # Number of batches to use for validation
        model_path=None,        # Path to save model checkpoints
        module_name="reasoning",
        epsilon=0.2,            # PPO clipping parameter
        beta=0.04               # KL penalty coefficient
):
    # Store the current policy as the old policy
    model.train()

    train_layers = get_training_layers(model)
    
    global_step = 0
    for epoch in range(config.num_epochs):
        idx = 0
        epoch_loss = 0.0
        for item_idx, question_idx, answer_idx in enumerate(train_loader):
            if idx % group_size == 0:
                optimizer.zero_grad()
            idx += 1

            output_idx = chat(model, tokenizer, question_idx, item_idx, config)

            output_text = tokenizer.decode(output_idx, skip_special_tokens=True)
            reward, result = reward_fn(question_idx, output_text)
            policy_losses_sum = torch.tensor(0.0).to(model.device)
            for t in range(len(output)):
                # Get log probability from current policy
                current_log_prob = model.log_prob(output[t] | question, output[:t])

                # Direct policy loss
                policy_loss = -current_log_prob * reward
                policy_losses_sum += policy_loss

            # Clipped surrogate objective
            policy_loss = policy_losses_sum / len(output)

            # KL penalty is not applied since limited resources but modify only few layers to avoid catastrophic forgetting
            loss = policy_loss

            # Update policy model
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_layers, max_norm=0.1)

            if idx % group_size == 0:
                optimizer.step()
                optimizer.zero_grad()

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
            val_loss_sum = 0.0
            pass_count = 0
            fail_count = 0
            sample_count = 0
            recent_failures = []
            with torch.no_grad():
                for idx, question in enumerate(train_loader):
                    output = chat(model, tokenizer, question, idx, config)
                    reward = reward_fn(question, output)

                    question_text = tokenizer.decode(question[0], skip_special_tokens=True)
                    output_text = tokenizer.decode(output, skip_special_tokens=True)

                    if reward.item() > 1.0:
                        pass_count += 1
                    else:
                        fail_count += 1
                        recent_failures.append({
                            'Test Target': question_text,
                            'Generated': output_text,
                            'Result': reward.item()
                        })
                    sample_count += 1
                    val_loss_sum += reward.item()
                    # Limit evaluation to a few batches for speed
                    if sample_count >= validation_batches:
                        break
            avg_val_loss = val_loss_sum / validation_batches
            pass_rate = pass_count / sample_count if sample_count > 0 else 0
            writer.add_scalar("Validation/Loss", avg_val_loss, epoch)
            writer.add_scalar("Reasoning/PassRate", pass_rate, epoch)
            print(f"Epoch {epoch} validation loss: {avg_val_loss:.4f}, Reasoning Pass Rate: {pass_rate:.4f}")

            # Log recent failures/errors in TensorBoard (only a few samples)
            for idx, failure in enumerate(recent_failures[-5:]):
                log_text = f"Test Target: {failure['Test Target']}\n" \
                           f"Generated: {failure['Generated']}\n" \
                           f"Expected: {failure['Expected']}\n" \
                           f"Result: {failure['Result']}"
                writer.add_text(f"Validation/Failure_{idx}", log_text, epoch)
            model.train()

            config.save()
    return model