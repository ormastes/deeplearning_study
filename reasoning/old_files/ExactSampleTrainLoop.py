import torch
import os
from transformers import AdamW
import torch.nn.functional as F

def train_exact(
        model,           # The model being trained
        optimizer,       # Optimizer for training
        tokenizer,       # Tokenizer for encoding/decoding
        config,
        writer,
        chat,                   # Function to generate output
        train_loader,           # Training questions
        reward_fn = None,              # Function to calculate rewards
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
    train_layers = config.get_training_layers(model)
    global_step = 0
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        for idx, qa in enumerate(train_loader):
            qa = qa.to(model.device)
            cur_in = qa[:, start:end]
            lengths = [qa[i].shape[0] for i in range(len(qa))]
            cur_in = qa[:, start:end]
            max_len = max(lengths)
            for i in range(max_len-1):
                current_context = min(config.context_len, i + 1)
                start = i - current_context + 1  # so that the window has 'current_context' tokens
                end = i + 1  # input window is [start, end) => length = current_context
                context_size = end - start

                cur_in = qa[:, start:end]
                cur_mask = torch.ones_like(cur_in).to(model.device)
                expected = qa[:, i + 1]

                logits = model(input_ids=cur_in, attention_mask=cur_mask).logits

                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), expected.flatten())

                input_text = tokenizer.decode(qa[0], skip_special_tokens=True)
                out_text = tokenizer.decode(max_indices[0], skip_special_tokens=True)


                # get last logits
                last_logit = logits[..., -1, :].contiguous()
                last_logit = last_logit.clamp(min=-100, max=100)

                # Flatten the tokens for loss computation
                loss = F.cross_entropy(
                    last_logit,
                    expected,
                    ignore_index=tokenizer.pad_token_id  # ignore pad tokens if applicable
                )


                if torch.isnan(loss):
                    print("Loss is NaN!")
                    input_text = tokenizer.decode(question_ids[0], skip_special_tokens=True)
                    print(f"Input text: {input_text}")
                    chars_out_onehot = torch.nn.functional.one_hot(last_logit, num_classes=tokenizer.vocab_size)
                    chars_out = torch.argmax(chars_out_onehot, dim=-1)
                    output_text = tokenizer.decode(chars_out[0], skip_special_tokens=True)
                    print(f"Output text: {output_text}")
                    print(f"Answer text: {tokenizer.decode(answer_ids[0], skip_special_tokens=True)}")
                    continue

                loss_factor = max(1.0, context_size / 12)
                loss = loss * loss_factor
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