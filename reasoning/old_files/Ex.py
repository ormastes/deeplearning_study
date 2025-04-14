import os
import copy
import json
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from transformers.optimization import Adafactor
from datasets import Dataset

# For hyperparameter optimization
import optuna
import pickle
from Config import SimpleConfig
from ClangReplInterface import ClangReplInterface

ref_checkpoint_path = "./saved_models/sample/checkpoint.pt"
last_checkpoint_path = "./saved_models/reasoning/checkpoint.pt"
checkpoint_dir_pre = "./saved_models/reasoning/epoch_"

test_target_object_file = "./convert/test_target_and_object/converted_data.json"

config = SimpleConfig()

clang_repl = ClangReplInterface()

# num_iterations=1, num_steps=500, batch_size=4, num_generations=4, max_completion_length=128, kl=0.1,
# learning_rate=5e-6, mu=3, epsilon=0.2,
#
# lr: 7.205691481165551e-05 kl_lambda: 0.2654706177039008 epsilon: 0.019437902361559744 num_grpo: 1
# lr: 1.1111588431283189e-06 kl_lambda: 0.15842765249477542 epsilon: 0.11144786260484413 num_grpo: 3
is_finding_opt = False
if not is_finding_opt:
    num_epochs = 200
    lr = 1.1111588431283189e-06
    kl_lambda = 0.15842765249477542
    epsilon = 0.11144786260484413
    num_grpo = 1
    save_epochs = 10


def reward_atag(front, end, response):
    start = response.find(front + end)
    end = response.find(front + '/' + end)
    reward = 0
    if start != -1: reward += 0.1
    if end != -1: reward += 0.1
    tag_len = len(front + end)
    if start + tag_len < end:
        if len(response[start + tag_len:end].strip()) > 1:
            reward += 0.1


def reward_correct(response):
    # handle only first answer
    reward = 0.0
    start = response.find("<Clang-repl Test>")
    end = response.find("</Clang-repl Test>")
    tag_len = len("<Clang-repl Test>")
    if start != -1 and end != -1 and start + tag_len < end:
        test = response[start + tag_len:end].strip()
        result = clang_repl.run_verify(test)
        reward = 0.0
        if result == 'ok':
            reward = 2.0
        elif result == 'fail':
            reward = 1.0
        elif result == 'error':
            reward = 0.0
        else:
            assert False
        return reward
    else:
        return reward


def reward(completions):
    # https://blog.gopenai.com/coding-grpo-from-scratch-a-guide-to-distributed-implementation-with-qwen2-5-1-5b-instruct-59b34227edacabs
    responses = [completion[0]['content'] for completion in completions]
    format_rewards = []
    rewards = []
    for response in responses:
        score = 0.0
        score += reward_atag("<", "Test Object>", response)
        score += reward_atag("<", "Input Data>", response)
        score += reward_atag("<", "Expected Output>", response)
        score += reward_atag("<", "Clang-repl Test>", response)
        score += reward_atag("[", "REASON]", response) * 2
        score += reward_atag("[", "ANSWER]", response) * 2
        score = score / 10
        format_rewards.append(score)
        correct_reward = reward_correct(response)
        rewards(score + correct_reward)


def object_hiper_param(trial):
    # Shortened training for demonstration:
    num_epochs = 1  # 2   # or 2–3, to save time during hyperparameter search

    # Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    kl_lambda = trial.suggest_float("kl_lambda", 0.0, 1.0)
    epsilon = trial.suggest_float("epsilon", 0.01, 0.2)
    num_grpo = trial.suggest_int("num_grpo", 1, 3, step=1)

    return num_epochs, lr, kl_lambda, epsilon, num_grpo


def print_memory(tag):
    # Make sure you have a GPU device available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Print current allocated and reserved memory in MB:
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    print(tag)
    print(f"Memory allocated: {allocated:.2f} MB")
    print(f"Memory reserved: {reserved:.2f} MB")


def samping(model, tokenizer, device, epoch, writer, sample_prompt, expected):
    # Include attention_mask in the tokenization
    sample_prompt = f"### Instruction\n\n{sample_prompt}\n\n### Response"
    inputs = tokenizer(sample_prompt, return_tensors="pt", return_attention_mask=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Pass the attention_mask and explicitly set pad_token_id to eos_token_id for reliable generation
    generated_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=20,
        pad_token_id=tokenizer.eos_token_id
    )
    sample_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    sample_text = sample_text.strip()
    print(f"Sample Output (Epoch {epoch + 1}): {sample_text}")
    print("Expected:", expected)
    writer.add_text("Sample Output", f"Epoch {epoch + 1}: {sample_text}", epoch)


def selective_log_softmax(logits, input_ids):
    # https://blog.gopenai.com/coding-grpo-from-scratch-a-guide-to-distributed-implementation-with-qwen2-5-1-5b-instruct-59b34227edac
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def add_front_transformer_block(self, copy_weights: bool = True):
    # Retrieve the current first transformer block.
    layer_index = 1
    original_first_block = self.model.layers[layer_index]

    # Create a new block.
    new_block = copy.deepcopy(original_first_block) if copy_weights else type(original_first_block)()

    self.model.layers.insert(layer_index, new_block)

    self.config.num_hidden_layers += 1


def get_layer_params(self, layer_index: int = 0):
    first_params = list(self.model.layers[0].parameters())
    sec_params = list(self.model.layers[0].parameters())
    # last_params = list(self.model.layers[-1].parameters())
    return first_params + sec_params  # + last_params


def gen_logits(model, batch):
    # Assume the batch contains input_ids and attention_mask.
    prompt_ids = batch['input_ids']
    prompt_length = prompt_ids.shape[1]

    # Generate new tokens.
    output = model(**batch)

    logits = output.logits[:, :, :]
    ids = torch.argmax(logits, dim=-1)

    # Extract the generated completion tokens (i.e. tokens after the prompt).
    return logits[:, prompt_length:], ids[:, prompt_length:]


# ------------------------------------------------
# Load Q&A from JSON file (manual_data_set/QA.json)
# and create a list of {"content": "..."}
# ------------------------------------------------
def load_qa_dataset(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_examples = []
    for item in data:
        q = item.get("Q", "")
        a = item.get("A", "")
        content = f"### Instruction\n\n{q}\n### Response\n\n{a}\n"
        train_examples.append({"content": content + "<|endoftext|>"})
    return train_examples


def load_sample_dataset(pk_file):
    with open(config.dataset_file, "rb") as f:
        global_samples = pickle.load(f)
        sample_dataset = []
        for sample in global_samples:
            sample_dataset.append({"content": sample + "<|endoftext|>"})
        return sample_dataset


def load_reasoning_dataset(pk_file):
    with open(test_target_object_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        sample_dataset = []
        for sample in data["train"]:
            content = f"### Instruction\n\nn<Test Target Object>\n{sample["Test Target Object"]}\n<Test Target Object>\nWrtie a Clang-repl Test\n### Response\n"
            sample_dataset.append({"content": content})
        return sample_dataset

# Provide the path to your Q&A JSON file
qa_json_path = "manual_data_set/QA.json"
train_data_prompt = load_qa_dataset(qa_json_path)
train_data_sample = load_sample_dataset(config.dataset_file)
train_data = train_data_prompt + (train_data_sample * 10)

# Create a Hugging Face Dataset from the list
train_dataset = Dataset.from_list(train_data)

# ------------------------------------------------
# Define Tokenization
# ------------------------------------------------
model_id = "bigcode/starcoder2-3b"
# Load tokenizer from saved directory if exists; otherwise, load from pretrained.
tokenizer_save_dir = "./saved_models/tokenizer"
if os.path.exists(tokenizer_save_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_dir)
    print("Loaded tokenizer from saved checkpoint.")
else:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples["content"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

print("eos: ", tokenizer.eos_token, tokenizer.eos_token_id)

tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["content"])
tokenized_dataset.set_format("torch")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# ------------------------------------------------
# Define Training Function
# ------------------------------------------------
def train_and_evaluate(
        model,
        ref_model,
        dataloader,
        optimizer,
        device,
        num_epochs,
        num_grpo,
        epsilon,
        kl_lambda,
        scaler,
        save_epochs,
        start_epoch
):
    """
    Train the model for `num_epochs` with `num_grpo` PPO groups each epoch,
    and return a metric (e.g., final average loss).
    """
    # Initialize TensorBoard writer (optional)
    writer = SummaryWriter(log_dir="runs/starcoder2_reasoning")

    # --- Generate sample output text after each epoch ---
    model.eval()  # Switch to eval mode for generation
    with torch.no_grad():
        samping(model, tokenizer, device, 0, writer, "In Custom Clang-repl, What is the prompt in Custom Clang-repl?",
                "```\n>>> (prompt)\n```")
        samping(model, tokenizer, device, 0, writer,
                "In Custom Clang-repl, Do we allow multiline comments or backslash-extended lines in Custom Clang-repl Test?",
                "Custom Clang-repl takes only one line input.")
    model.train()  # Switch back to training mode

    global_step = 0
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0

        old_model = None
        old_model = copy.deepcopy(model)
        old_model = old_model.half()
        old_model.eval()
        for param in old_model.parameters():
            param.requires_grad = False

        for grpo_idx in range(num_grpo):
            for step, batch in enumerate(dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    completion_logits, completion_ids = gen_logits(model, batch)

                    # Decode the token ids to text strings.
                    output_text_lists = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                    advantages = torch.tensor(
                        reward(output_text_lists),
                        dtype=torch.float32,
                        device=device
                    )

                    mean_rewards = advantages.mean()  # .repeat_interleave(num_generations)
                    std_rewards = advantages.std()  # .repeat_interleave(num_generations)

                    A_hat = (advantages - mean_rewards) / std_rewards

                    # old model forward
                    with torch.no_grad():
                        old_completion_logits, old_completion_ids = gen_logits(old_model, batch)

                    # reference model forward
                    with torch.no_grad():
                        ref_completion_logits = gen_logits(ref_model, batch)
                        ref_outputs, _ = ref_model(**batch)

                    input_ids = batch["input_ids"]

                    # Probability ratio
                    model_log_logits = selective_log_softmax(completion_logits, completion_ids)
                    old_model_log_logits = selective_log_softmax(old_completion_logits, old_completion_ids)
                    probability_ratio = torch.exp(model_log_logits - old_model_log_logits)

                    # Unclipped objective
                    unclipped_objective = probability_ratio * A_hat

                    # Clipped objective
                    clipped_ratio = torch.clamp(probability_ratio, 1 - epsilon, 1 + epsilon)
                    clipped_objective = clipped_ratio * A_hat

                    # ppo_loss = clipped_objective.mean()
                    ppo_loss = -torch.min(unclipped_objective, clipped_objective).mean()
                    # ppo_loss = loss.mean()

                    # KL
                    model_log_probs = F.log_softmax(completion_logits, dim=-1)
                    ref_log_probs = F.softmax(ref_completion_logits, dim=-1)
                    kl_div = F.kl_div(model_log_probs, ref_log_probs, reduction='batchmean')

                    combined_loss = ppo_loss + kl_lambda * kl_div

                scaler.scale(combined_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += combined_loss.item()

                # TensorBoard logging
                writer.add_scalar("Loss/combined_loss", combined_loss.item(), global_step)
                writer.add_scalar("Loss/ppo_loss", ppo_loss.item(), global_step)
                writer.add_scalar("Loss/kl_div", kl_div.item(), global_step)
                #writer.add_scalar("Loss/original_loss", loss.item(), global_step)

                global_step += 1

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
        writer.add_scalar("Epoch/Average_Loss", avg_loss, epoch + 1)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1
        }
        torch.save(checkpoint, last_checkpoint_path)

        if save_epochs is not None and epoch % save_epochs == 0:
            global checkpoint_dir_pre
            checkpoint_dir = checkpoint_dir_pre + str(epoch + 1)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1
            }
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")

            # Save tokenizer only if it has not been saved before.
            tokenizer_save_dir = "./saved_models/tokenizer"
            if not os.path.exists(tokenizer_save_dir):
                os.makedirs(tokenizer_save_dir, exist_ok=True)
                tokenizer.save_pretrained(tokenizer_save_dir)
                print("Tokenizer saved.")

            # --- Generate sample output text after each epoch ---
            model.eval()  # Switch to eval mode for generation
            with torch.no_grad():
                samping(model, tokenizer, device, epoch, writer,
                        "In Custom Clang-repl, What is the prompt in Custom Clang-repl?", "```\n>>> (prompt)\n```")
                samping(model, tokenizer, device, epoch, writer,
                        "In Custom Clang-repl, Do we allow multiline comments or backslash-extended lines in Custom Clang-repl Test?",
                        "Custom Clang-repl takes only one line input.")
                samping(model, tokenizer, device, epoch, writer, "Make python string reverse function",
                        "def reverse(text):\n    return reverse(text[1:])+text[0]")
                samping(model, tokenizer, device, epoch, writer,
                        "<Test Target Object>\nAdd two integers. and return the sum.\n</Test Target Object>\n", "....")
                samping(model, tokenizer, device, epoch, writer,
                        "<Test Target>\nbool isEven(int x) {\n    return (x % 2) == 0;\n}\n</Test Target>\n", "....")
                print(
                    "=====================================================================================================")
            model.train()  # Switch back to training mode

    writer.close()

    # Return final average loss as the metric to minimize
    return avg_loss


def train(
        num_epochs,
        lr,
        kl_lambda,
        epsilon,
        num_grpo,
        save_epochs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if a latest checkpoint exists to load model and optimizer states
    if os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(last_checkpoint_path, map_location=torch.device("cpu"))
        _model = AutoModelForCausalLM.from_pretrained(model_id)
        config = _model.config
        config.num_hidden_layers += 2
        model = AutoModelForCausalLM.from_config(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer = Adafactor(get_layer_params(model), lr=lr, relative_step=False, scale_parameter=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from {last_checkpoint_path} at epoch {start_epoch}")
    else:
        if os.path.exists(ref_checkpoint_path):
            print_memory(1)
            checkpoint = torch.load(ref_checkpoint_path, map_location=torch.device("cpu"))
            _model = AutoModelForCausalLM.from_pretrained(model_id)
            config = copy.deepcopy(_model.config)
            config.num_hidden_layers += 2
            model = AutoModelForCausalLM.from_config(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            optimizer = Adafactor(get_layer_params(model), lr=lr, relative_step=False, scale_parameter=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"Loaded checkpoint from {ref_checkpoint_path} at epoch {start_epoch}")
            print_memory(7)

        else:
            assert (False, "prompt_last_checkpoint_path must exist")

    # Clear cached memory that is no longer used
    torch.cuda.empty_cache()
    gc.collect()
    print_memory(9)

    # Reference model (for KL)
    old_model = None
    ref_model = copy.deepcopy(model).half().eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # DataLoader
    batch_size = 1
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    # AMP GradScaler
    scaler = torch.cuda.amp.GradScaler()

    # Train & get final metric
    final_avg_loss = train_and_evaluate(
        model=model,
        ref_model=ref_model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        num_grpo=num_grpo,
        epsilon=epsilon,
        kl_lambda=kl_lambda,
        scaler=scaler,
        save_epochs=save_epochs,
        start_epoch=start_epoch
    )

    # Return the final average loss to Optuna
    return final_avg_loss


# ------------------------------------------------
# Optuna Objective Function
# ------------------------------------------------
def objective(trial):
    """
    Defines how Optuna will run each trial:
    - sample hyperparameters
    - set up the model & optimizer with those
    - run a short training loop
    - return a metric (the final avg loss) to minimize
    """
    num_epochs, lr, kl_lambda, epsilon, num_grpo = object_hiper_param(trial)

    print(
        f"[Optuna] Trial hyperparameters -> lr: {lr}, kl_lambda: {kl_lambda}, epsilon: {epsilon}, num_grpo: {num_grpo}")
    return train(
        num_epochs=num_epochs,
        lr=lr,
        kl_lambda=kl_lambda,
        epsilon=epsilon,
        num_grpo=num_grpo)


# ------------------------------------------------
# Run Optuna Study
# ------------------------------------------------
if __name__ == "__main__":
    if is_finding_opt:
        # Create study to minimize final loss
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)  # You can increase n_trials

        print("Study completed!")
        print("Best trial:")
        best_trial = study.best_trial
        print(f"  Value: {best_trial.value}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"#    {key}: {value}")
        # Study completed!
        # Best trial:
        #  Value: 715.3611988491482
        #  Params:
        #    lr: 0.0002746775018590349
        #    kl_lambda: 0.10527608699361579
        #    epsilon: 0.12442505216944565
        #    num_grpo: 2
    else:
        train(
            num_epochs=num_epochs,
            lr=lr,
            kl_lambda=kl_lambda,
            epsilon=epsilon,
            num_grpo=num_grpo,
            save_epochs=save_epochs
        )
