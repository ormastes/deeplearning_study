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

### Equation details

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

**Sample Range:**
when $\theta$ is 0.2

$L_i(\theta)\in(0, 1.2 A^{\pi_{\theta_t}}(s,a_i)]\text{ if } A^{\pi_{\theta_t}}(s,a_i) > 0$

$L_i(\theta)\in(0, 0.8 A^{\pi_{\theta_t}}(s,a_i)]\text{ if } A^{\pi_{\theta_t}}(s,a_i) < 0$

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


## Technical difficulty
Since cuda has a lot of bug which is much better than AMD product though. I have rollback may cuda driver version with last year cuda driver and used full GPU base running.
ChatGPT gave me a lot of help and also a lot of buggy codes. To match align between ids and logits, I added an offset to logits proposely but ChatGPT point out it is wrong code and does not need align but I was right. It give some code snifit which is match to GRPO equations and the code has comment what output dimension of tensor would be. It was lie and until I print it out I believed it. It remove critical paramters when I request update in function to add some functionality. I found it API document.
Too small gpu memory, even for simple LLM training 49GB VRAM is too small to try something. I should find line by line memory in efficient code. I started to train 7B model to train but finally I used 3B model for LLM training with partial layer training. Because of optimizer and illregular structure of current LLM models require much more memory for training than expected.
Often even well known codes have bug equations. With a lot of computing power and sample may not effect final result but with limited resource and low precision, these small mistake effect final result.


## Lesson
Fully understand equeations and why and how they should applied to training. 
Do not trust ChatGPT. It often gave you answer with fake evidence. Run it, Log it and check it.
Use hand crafted data not wild existing data.



## Helpful materials
[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
[Coding GRPO from Scratch: A Guide to Distributed Implementation with QWen2.5–1.5B-Instruct](https://blog.gopenai.com/coding-grpo-from-scratch-a-guide-to-distributed-implementation-with-qwen2-5-1-5b-instruct-59b34227edac)
[Introducing the Clipped Surrogate Objective Function](https://huggingface.co/learn/deep-rl-course/en/unit8/clipped-surrogate-objective)
First of all, just understanding it and implement it was very different. I thought I understand it but it does not work and I find what I didn't understand actually when I am implementing it. "Coding GRPO from Scratch" was very helpful whenever I met some difficulty and I was able to find how they solved it.

