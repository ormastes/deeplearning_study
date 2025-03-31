m# LLM Reasoning with Programming

## Key concept of LLM training for reasoning
A lot of methods have been applied to LLM for reasoning. One of them are chain of thought which made great performance gain but not enough to reach reasoning capabilty.
OpenAI developed some training method for reasoning. From the hint of OpenAI which highly base on reinforcment learning, Deepseek anounced how to train LLM model for reasoning.
Reasoning training does not follow predefined training data but it find a path to answer by itself with try and error.
Although Deepseek does not open everything but can take outline of there training. They made an methmatical questions which can be calculated by computer. During training, methmatical questions (may with some value changes) are questioned and take response. If the response has enough chain of thouhgt or reasoning and the parsed answer value correctness is applied to training.

## Project outline
Deepseek used methatics for training. however, here I used a C/C++ Programming in specific test case genration.
Repl(Read-Eval-Print Loop) can be used as a programming play ground. Can easily make a test case writing and verification environment.
I didn't used Python Repl environment but used clang repl which C/C++ compiler based repl environment because LLM can generate good quality of python code and it is very hard to distinguish improvement since it is already too good.
However, C/C++ code generation from LLM is bad enough even if it is very big model. It is easy to verify how reasoning is effective for training for programming. 
More than that I want to make a LLM to generate simple test case for cdoctest which I made for C/C++ unit test. It is duplication of Python doctest (https://docs.python.org/3/library/doctest.html) which python repl can be embedded on comment of python code. The code in python document is runnable as testcase in both IDE and CLI. I have made cdoctest for both windows and linux platform and runnable from vscode IDE with my extension for it. From the extension, I will add test case genration command for cdoctest.
Current status, LLM seems working and few jobs to integrated with my cdoctest vscode extension which may takes couple of months.

## Training data
I have used data from
1. CPP-UT-Bench:
   https://huggingface.co/datasets/Nutanix/CPP-UNITTEST-BENCH
2. CompCodeVet: Compiler-Validated Data Curation
   https://huggingface.co/datasets/Elfsong/Mercury
   https://github.com/Elfsong/Mercury/blob/main/src/dpo_train.py
   https://github.com/Elfsong/Mercury/blob/main/src/sft_train.py
3. <strike>CITYWALK: Enhancing LLM-Based C++ Unit Test Generation
   https://zenodo.org/records/14022506 </strike><< not a dataset.
   But the logic can be used for sample data, and training data refine.
but they too big and it is dirty raw data. So, I have made raw training data and validation data with chat gpt and evaluate contents by eyes and running it with script that its syntax is valid.

Whenever make a data to training, check training format or data which the model used. And use same format. it will reduce error and work much more well.

### Prompt training(fine tunning)
To prevent too much effort for start up, I just add couple of layers and train those layer even I have only limited resource. 
The layers place on front because front layer can steering overall results.
I have first made prompts for generating test cases and check it can actually work. then made it for QnA training dataset to memorize the prompt rules in the model.

### Training reasoning
I have prepared about nine categories of data. each category has 15 items. 14 will be used for training while 1 will be used for validation.
['simple arithmetic', 'simple if', 'simple loop', 'loop and if', 'simple state', 'recursive function', 'pointer manipulation', 'string manipulation', 'sort algorithm']

## Equations
Both **GRPO** and **PPO** share the same basic idea of using a clipped surrogate objective to constrain policy updates, but **GRPO** adapts the formulation specifically for LLM fine‐tuning by replacing the per‐token advantage computed with a critic (value network) with a group‐relative advantage computed from multiple sampled completions per prompt.

---

### PPO Objective

In **PPO** the objective is usually written as

$$
J_{PPO}(\theta) = \mathbb{E}\Bigg[\min\Bigg(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\,A_t,\; \text{clip}\Big(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)},\,1-\epsilon,\,1+\epsilon\Big) A_t\Bigg)\Bigg],
$$

where:

- $$ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} $$ 
  is the probability ratio,
- $$ A_t $$
  is the advantage estimated typically via Generalized Advantage Estimation (GAE) using a learned value function,
- $$\epsilon$$ 
  is the clipping parameter to keep updates "proximal."

This formulation requires maintaining a separate value network (critic) to compute $$A_t$$ at each timestep, which can be challenging for LLMs when rewards are sparse or only provided at the end of the generated sequence.

---

### GRPO Objective and Its Differences

In **GRPO** (Group Relative Policy Optimization), the core idea is to remove the need for an extra value network by computing a group‐relative advantage from multiple responses generated for the same prompt with simple evaluator (which is not a neural network). Concretely, for a given prompt $$p$$, you sample a group of $$G$$ responses and obtain their rewards $$\{r_1, r_2, \dots, r_G\}$$. Then, the advantage for each sample is calculated as

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}.
$$

In other words, find how much does it better than others. 

The **GRPO** objective then becomes

$$
J_{GRPO}(\theta) = \frac{1}{G}\sum_{i=1}^G \min\!\Bigg(\frac{\pi_\theta(r_i|p)}{\pi_{\theta_{old}}(r_i|p)}\,\hat{A}_i,\; \text{clip}\!\Big(\frac{\pi_\theta(r_i|p)}{\pi_{\theta_{old}}(r_i|p)},\,1-\epsilon,\,1+\epsilon\Big)\hat{A}_i\Bigg) - \beta\, D_{KL}\big(\pi_\theta \,\|\, \pi_{ref}\big).
$$

Apply adventage to grouped ppo for normalize output rather just one output and to prevent catastrophic forgetting use KL regulization 

Key differences compared to **PPO** include:

- **Advantage Estimation:**  
  - *PPO* computes $$A_t$$ at each token using a critic (value network) over a trajectory.
  - *GRPO* computes a single advantage per generated response by normalizing its reward relative to the group’s mean and standard deviation. This is especially useful in LLMs where the reward (e.g. for correctness or formatting) is often provided only at the end of the sequence.  

- **No Extra Value Network:**  
  - By deriving the advantage directly from group rewards, **GRPO** eliminates the need for a separate value function. This simplifies the training pipeline and reduces memory and computational overhead.  

- **KL Regularization:**  
  - Both methods incorporate a KL divergence term to keep the updated policy close to a reference policy. In **GRPO**, this is integrated directly into the loss rather than into the reward signal, maintaining stability without complicating the advantage estimation.  

---

### Equations detail

#### Reward and Advantage

The advantage is defined as:

$$
\hat{A}_i = \frac{r_i - \mu}{\sigma + \epsilon},
$$
where:
- $r_i$ is the reward for the $i$th response,
- $\mu$ is the mean reward for the group,
- $\sigma$ is the standard deviation of the rewards,
- $\epsilon$ is a small constant added for numerical stability.

**Theoretical Range:**

**Theoretically**, because $r_i - \mu$ can be any real number and $\sigma > 0$ (if the rewards are not all identical), the normalized advantage $\hat{A}$ can take any value in $(-\infty, +\infty)$.

**Practical Range:**

- In practice, when rewards are roughly normally distributed or normalized, $\hat{A}$ behaves like a z-score. For a normal distribution, about 99.7% of values lie within approximately $[-3, 3]$. Thus, you can expect most $\hat{A}$ values to be in the range of about $-3$ to $3$.

**Code:**
```python
print_logits_ids("model response", response_truncated_logits, respone_ids) # good format confirmed, response_truncated_logits: RESPONSE_LOGITS, respone_ids: RESPONSE_IDS
model_log_logits = selective_log_softmax(response_truncated_logits, respone_ids, tokenizer).check_shape(RESPONSE_IDS)
model_log_logits.log()
print_logits_ids("old model response", old_response_truncated_logits, respone_ids) # good format confirmed, old_response_truncated_logits: RESPONSE_LOGITS, respone_ids: RESPONSE_IDS
old_model_log_logits = selective_log_softmax(old_response_truncated_logits, respone_ids, tokenizer).check_shape(RESPONSE_IDS)
old_model_log_logits.log() 
probability_ratio = torch.exp(model_log_logits - old_model_log_logits).check_shape(RESPONSE_IDS)
probability_ratio.log() 

# Calculate mean and std per batch (along dim=1) and repeat to match original size
mean_rewards = advantages.mean(dim=1).repeat_interleave(group_size).check_shape([batch_size]).check_range(0, 2.24)
mean_rewards.log()
std_rewards = advantages.std(dim=1).repeat_interleave(group_size).check_shape([batch_size]).check_range(0, float('inf'))
std_rewards.log() 

# Reshape back to original form
advantages = advantages.view(-1)
advantages.check_shape([batch_size]).log("advantages before A_hat")
A_hat = ((advantages - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1).check_shape([batch_size, 1]) 
A_hat = torch.clamp(A_hat, -5, 5) # I have added a clipping since small sample cause too much variation.

# 5. grouped_ppo Loss Calc
print_step("5. Grouped ppo Loss Calc")            
# PPO objective calculations.
unclipped_objective = probability_ratio
unclipped_objective.check_shape(RESPONSE_IDS).log()
epsilon_high = torch.full_like(unclipped_objective, 1 + epsilon).check_shape(RESPONSE_IDS) # Instead clipping, I just put upper bound which result better ouput
_grouped_ppo_loss = - torch.minimum(unclipped_objective, epsilon_high)
_grouped_ppo_loss.check_shape(RESPONSE_IDS).log("before A_hat multiply")
_grouped_ppo_loss = _grouped_ppo_loss * A_hat
grouped_ppo_loss = _grouped_ppo_loss.mean(dim=-1).check_shape([batch_size])
grouped_ppo_loss.log() # sample epsilon=0.2
    
```

where the mean and variance over a batch of $G$ samples are given by

$$
\mu = \frac{1}{G} \sum_{i=1}^{G} r_i, \quad \sigma = \frac{1}{G} \sum_{i=1}^{G} (r_i - \mu)^2.
$$

#### Probability Ratio

The probability ratio is defined as:

$$
r_i(\theta) = \exp\left(\log \pi_\theta(r_i \mid p) - \log \pi_{\theta_{\text{old}}}(r_i \mid p)\right).
$$

Range: ${\pi_\theta(r_i \mid p)}\in(0,1]$

Range: $\log {\pi_\theta(r_i \mid p)}\in(−∞,0]$

#### After Exponentiation
The operation $r(\theta) = \exp(\Delta \ell)$ 

Range: $\Delta \ell \in (-\infty, +\infty)$ , $r(\theta) \in (0, \infty)$

- If $\Delta \ell = 0$, then $r(\theta) = \exp(0) = 1.$
- If $\Delta \ell > 0$, then (new policy assigns a higher probability) $r(\theta) > 1.$
- If $\Delta \ell < 0$, then $0 < r(\theta) < 1.$


which is equvalent to next equation. Previous one use log space for mathmetical stability.
$$
r_i(\theta) = \frac{\pi_{\theta_{\text{old}}}(r_i \mid p)}{\pi_\theta(r_i \mid p)}
$$


#### Grouped PPO Loss

The individual surrogate loss(PPO) is given by:

$$
L_i(\theta) = \min\Big( r_i(\theta) \, \hat{A}_i, \, \text{clip}\big(r_i(\theta),\, 1-\epsilon_{\text{clip}},\, 1+\epsilon_{\text{clip}}\big) \, \hat{A}_i \Big).
$$
This equation can cause misconception which just clip as it expressed but actual intent is like next.

$$
L_i(\theta) =
\begin{cases}
\min\Bigl(r_i(\theta) \, A^{\pi_{\theta_t}}(s,a_i),\, (1+\epsilon) \, A^{\pi_{\theta_t}}(s,a_i)\Bigr)
& \text{if } A^{\pi_{\theta_t}}(s,a_i) > 0, \\[1ex]
\max\Bigl(r_i(\theta) \, A^{\pi_{\theta_t}}(s,a_i),\, (1-\epsilon) \, A^{\pi_{\theta_t}}(s,a_i)\Bigr)
& \text{if } A^{\pi_{\theta_t}}(s,a_i) < 0.
\end{cases}
$$
PPO loss is
$$
\text{loss} = - L_i(\theta)
$$

**Theoretical Range:** 

$L_i(\theta)\in(0, (1+\epsilon) A^{\pi_{\theta_t}}(s,a_i)]\text{ if } A^{\pi_{\theta_t}}(s,a_i) > 0$

$L_i(\theta)\in(0, (1-\epsilon) A^{\pi_{\theta_t}}(s,a_i)]\text{ if } A^{\pi_{\theta_t}}(s,a_i) < 0$

(Shouldn't it be $(-1-\epsilon)$ rather $(1-\epsilon)$ and I still doesn't understand why even clipping is exsit on there. Only higher bound seems more efficient.)

**Code:**
```python
# 3. kl_div Loss Calc
print_step("3. kl_div Loss Calc")    
# Calculate token-level log probabilities.
model_log_probs = selective_log_softmax(full_shift_logits, full_shift_ids, tokenizer)
model_log_probs.log() # RESPONSE_IDS
ref_log_probs = selective_log_softmax(ref_full_shift_logits, full_shift_ids, tokenizer)
ref_log_probs.log() # RESPONSE_IDS

# Compute token-level KL divergence.
token_kl_div = F.kl_div(model_log_probs, ref_log_probs, reduction='none', log_target=True).check_shape(FULL_IDS) # it is not an ids but parts of logits content. the shape is just like ids)
token_kl_div.log()
kl_div = token_kl_div.mean(dim=-1).check_shape([batch_size])
kl_div.log() # average over tokens. range (0, infite) but for output of similar model. It is very small. sample: kl_div=0.09
```

References:
OpenAI Spinning Up in Deep RL, Proximal Policy Optimization. [Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

The overall surrogate loss(Grouped PPO) is computed as:

$$
L_{\text{grouped\_PPO}}(\theta) = \frac{1}{G} \sum_{i=1}^{G} L_i(\theta).
$$

#### KL Penalty

The KL penalty term is defined as:

$$
L_{KL}(\theta) = \beta \, D_{KL}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big).
$$

**Theoretical Range:** $L_{KL}(\theta)\in[0, \infty)$

**Practical Range:** In practice, KL values are often small (e.g., 0.01 to 0.1) when policies are similar.

#### Total Loss

The total loss function is a combination of the surrogate loss (Grouped PPO)  and the KL penalty:

$$
L_{\text{total}}(\theta) = L_{\text{grouped\_PPO}}(\theta) + L_{KL}(\theta).
$$

**Code:**
```python
# kl_lambda is a scaling factor for the KL term
_combined_loss = grouped_ppo_loss + kl_lambda * kl_div
_combined_loss.check_shape([batch_size]).log() 
combined_loss = _combined_loss.mean()
combined_loss.log() # []
```

#### Equation tweeks
I have added a clipping since small sample cause too much variation.
Instead clipping to the objective, I just put upper bound which result better ouput. 


## Technical difficulty
Since cuda has a lot of bug which is much better than AMD product though. I have rollback may cuda driver version with last year cuda driver and used full GPU base running.
ChatGPT gave me a lot of help and also a lot of buggy codes. To match align between ids and logits, I added an offset to logits proposely but ChatGPT point out it is wrong code and does not need align but I was right. It give some code snifit which is match to GRPO equations and the code has comment what output dimension of tensor would be. It was lie and until I print it out I believed it. It remove critical paramters when I request update in function to add some functionality. I found it API document.
Too small gpu memory, even for simple LLM training 49GB VRAM is too small to try something. I should find line by line memory in efficient code. I started to train 7B model to train but finally I used 3B model for LLM training with partial layer training. Because of optimizer and illregular structure of current LLM models require much more memory for training than expected.
Often even well known codes have bug equations. With a lot of computing power and sample may not effect final result but with limited resource and low precision, these small mistake effect final result.
One epoch take too long and too many hiper parameter to config. I can eliminate couple of range of hiper parameter with tools like optuna but still it takes too much time.

## Lesson
Fully understand equeations and why and how they should applied to training. 
Do not trust ChatGPT. It often gave you answer with fake evidence. Run it, Log it and check it.
Use hand crafted data not wild existing data.
Find prompt format the model used which is dramatically well working while even well known format is not working.



## Helpful materials
[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
[Coding GRPO from Scratch: A Guide to Distributed Implementation with QWen2.5–1.5B-Instruct](https://blog.gopenai.com/coding-grpo-from-scratch-a-guide-to-distributed-implementation-with-qwen2-5-1-5b-instruct-59b34227edac)
[Introducing the Clipped Surrogate Objective Function](https://huggingface.co/learn/deep-rl-course/en/unit8/clipped-surrogate-objective)
First of all, just understanding it and implement it was very different. I thought I understand it but it does not work and I find what I didn't understand actually when I am implementing it. "Coding GRPO from Scratch" was very helpful whenever I met some difficulty and I was able to find how they solved it.


## Appendix. How run models

Generate sequence and reward caculation start.
```python
# 1. Model forward pass for generation.
print_step("1. Model train")
with torch.no_grad():
    full_ids, truncated_ids, respone_ids, prompt_lengths = generate_ids(model, batch, tokenizer, temperature)
    full_text_lists = tokenizer.batch_decode(truncated_ids, skip_special_tokens=True)
    reward_work.reward(full_text_lists, writer, log_group, global_step)
    # Release unused tensors from generation.
    full_text_lists = None
```

Run model to train and take logits(_full_shift_logits)
```python
_full_shift_logits, response_truncated_logits, _ = compute_logits(model, full_ids, prompt_lengths, respone_ids, tokenizer) 
```

Take old model and reference models outputs
Old model for training reasoning.
Reference model to prevent catastrophic forgetting.
```python
# 2. Run legacy models (old and reference models).
print_step("2. Legacy Models Run")
with torch.no_grad():
    _, old_response_truncated_logits, _ = compute_logits(old_model, truncated_ids, prompt_lengths, respone_ids, tokenizer, detach_out=True)
    ref_full_shift_logits, _, _ = compute_logits(ref_model, full_ids, prompt_lengths, respone_ids, tokenizer, detach_out=True)
```

Generate sequence of tokens without training.
```python
def generate_ids(model, batch, tokenizer, temperature):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    eos_token_id = tokenizer.eos_token_id

    # Determine prompt length for each example in the batch based on the first occurrence of EOS.
    prompt_lengths = []
    for i in range(input_ids.size(0)):
        seq = input_ids[i]
        # Find indices where the token equals the eos_token_id.
        eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        # If there's at least one occurrence, use its index + 1 (if you want to include the EOS in the prompt).
        # Otherwise, fallback to the full sequence length.
        if eos_positions.numel() > 0:
            first_eos = eos_positions[0].item() + 1
        else:
            first_eos = seq.size(0)
        prompt_lengths.append(first_eos)
    
    print_memory("Prompt lengths per batch element: " + str(prompt_lengths))

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,  # assuming max_length is defined globally
        temperature=temperature,
        do_sample=True,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        output_scores=True,
        return_dict_in_generate=True
    )
    full_ids = output.sequences.detach()
    truncated_ids = cut_ids_on_eos_tensor(full_ids, tokenizer.eos_token_id)    
    respone_ids =  pad_sequence([truncated_ids[idx][p_len:] for idx, p_len in enumerate(prompt_lengths)],
                                               batch_first=True, padding_value=tokenizer.pad_token_id)
    truncated_ids = pad_sequence(truncated_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    full_ids = pad_sequence(full_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    output = None
    print_memory("full_ids.shape[-1]: " + str(full_ids.shape[-1]))
    return full_ids, truncated_ids, respone_ids, prompt_lengths
```

Generate logits of model to training
```python
def compute_logits(model, full_ids, prompt_lengths, respone_ids, tokenizer, detach_out=False):
    # Pad the list of full_ids to a whole tensor with shape (batch, max_seq_length)
    full_ids = pad_sequence(full_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(dtype=torch.int32)
    
    # Create an attention mask where non-pad tokens are 1 and pad tokens are 0
    full_ids_mask = (full_ids != tokenizer.pad_token_id).to(dtype=torch.int32, device=full_ids.device)
    
    # Compute logits for the whole padded tensor.
    logits = model(input_ids=full_ids, attention_mask=full_ids_mask, early_stop=False).logits
    
    truncated_response_ids_list = []
    truncated_response_logits_list = []
    batch_size = full_ids.size(0)
    
    for i in range(batch_size):
        p_len = prompt_lengths[i]
        # Determine the true sequence length (ignoring padding) for this batch element.
        actual_length = full_ids_mask[i].sum().item()
        # Ensure prompt length does not exceed actual length.
        if p_len > actual_length:
            p_len = actual_length

        # Extract completion token IDs for this example.
        comp_ids = full_ids[i, p_len:actual_length].detach()
        # For logits, if you want to include the token just before the completion, slice from p_len-1.
        comp_logits = logits[i, p_len-1:actual_length-1, :]
        
        # Optionally, adjust lengths to be consistent (if needed by downstream code)
        #comp_ids, comp_logits = cut_tensors_by_min(comp_ids, comp_logits, 0)
        expected_len = respone_ids.shape[1]
        truncated_response_ids_list.append((comp_ids.detach() if detach_out else comp_ids)[:expected_len])
        truncated_response_logits_list.append((comp_logits.detach() if detach_out else comp_logits)[:expected_len, :])

    truncated_response_logits = pad_sequence(truncated_response_logits_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    truncated_response_ids = pad_sequence(truncated_response_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    return logits, truncated_response_logits, truncated_response_ids
```
