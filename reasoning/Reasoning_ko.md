# LLM Reasoning with Programming

## LLM Reasoning 훈련의 핵심 개념

대규모 언어 모델(LLM)은 reasoning 능력을 향상시키기 위해 다양한 방법으로 개선되어 왔습니다. Chain of thought 프롬프팅이 상당한 성능 향상을 보여주었지만, 진정한 reasoning 능력을 달성하기에는 충분하지 않았습니다.

OpenAI는 강화학습에 크게 의존하는 reasoning을 위한 고급 훈련 방법을 개발했습니다. 이에 따라 Deepseek는 최근 LLM 모델의 reasoning 훈련 접근 방식을 발표했습니다.

Reasoning 훈련의 핵심 혁신은 미리 정의된 훈련 데이터 경로를 따르지 않는다는 것입니다. 대신, 모델이 시행착오를 통해 답을 찾을 수 있도록 합니다. Deepseek가 모든 세부 사항을 공개하지는 않았지만, 그들의 훈련 접근 방식을 다음과 같이 요약할 수 있습니다:

1. 프로그래밍 방식으로 계산할 수 있는 수학 문제를 만들었습니다
2. 훈련하는 동안 이러한 문제(때로는 값이 수정된)가 모델에 제시됩니다
3. 모델의 응답은 다음을 기준으로 평가됩니다:
   - Chain of thought 또는 reasoning 과정의 충분성
   - 최종 구문 분석된 답변의 정확성

## 프로젝트 개요

Deepseek가 훈련을 위해 수학에 초점을 맞춘 반면, 이 프로젝트는 특정 테스트 케이스 생성과 함께 C/C++ 프로그래밍을 사용합니다.

REPL(Read-Eval-Print Loop)은 테스트 케이스를 생성하고 검증하기 위한 이상적인 프로그래밍 환경을 제공합니다. Python REPL(LLM이 이미 잘 수행하는 곳)을 사용하는 대신, 이 프로젝트는 C/C++ 프로그래밍을 위한 Clang 기반 REPL을 채택했습니다. 그 이유는 다음과 같습니다:

- LLM은 이미 고품질의 Python 코드를 생성합니다
- 대형 모델에서도 LLM의 C/C++ 코드 생성은 훨씬 더 취약합니다
- 이는 프로그래밍을 위한 reasoning 기반 훈련의 효과를 검증하기 쉽게 만듭니다

최종 목표는 cdoctest(Python의 doctest에서 영감을 받은 C/C++ 단위 테스팅 도구)를 위한 간단한 테스트 케이스를 생성할 수 있는 LLM을 만드는 것입니다. Cdoctest는 주석에 C/C++ REPL 코드를 내장하여 IDE와 CLI 환경 모두에서 테스트 케이스로 실행할 수 있게 합니다.

이 프로젝트는 Windows와 Linux 플랫폼 모두를 위한 cdoctest를 만들었으며, VS Code 확장 프로그램도 개발했습니다. Cdoctest를 위한 테스트 케이스 생성 명령이 이 확장 프로그램에 추가될 예정이며, 통합은 몇 개월 내에 완료될 것으로 예상됩니다.

## 훈련 데이터

프로젝트의 데이터 소스는 다음과 같습니다:

1. **CPP-UT-Bench**:  
   https://huggingface.co/datasets/Nutanix/CPP-UNITTEST-BENCH

2. **CompCodeVet: Compiler-Validated Data Curation**:  
   https://huggingface.co/datasets/Elfsong/Mercury  
   https://github.com/Elfsong/Mercury/blob/main/src/dpo_train.py  
   https://github.com/Elfsong/Mercury/blob/main/src/sft_train.py

3. ~~**CITYWALK: Enhancing LLM-Based C++ Unit Test Generation**~~:  
   ~~https://zenodo.org/records/14022506~~ (데이터셋이 아니지만, 그 로직은 샘플 데이터 및 훈련 데이터 정제에 사용될 수 있습니다)

원본 데이터셋은 크고 정리되지 않아서, ChatGPT를 사용하여 맞춤형 훈련 및 검증 데이터를 만들고, 내용의 유효성과 구문의 정확성을 스크립트 테스트를 통해 수동으로 평가했습니다.

훈련용 데이터를 준비할 때, 오류를 줄이고 결과를 개선하기 위해 모델이 훈련된 것과 동일한 형식을 사용하는 것이 중요합니다.

### 프롬프트 훈련(미세 조정)

초기 노력을 최소화하기 위해, 이 접근 방식은 모델 앞부분에 몇 개의 레이어를 추가하고 제한된 리소스에도 불구하고 해당 레이어만 훈련합니다. 전체 결과를 효과적으로 조정할 수 있기 때문에 앞쪽 레이어가 선택되었습니다.

이 과정은 테스트 케이스 생성을 위한 프롬프트를 만들고 그 기능을 검증하는 것으로 시작되었습니다. 그런 다음 이를 질문-답변 훈련 데이터셋으로 변환하여 프롬프트 규칙을 모델에 내장했습니다.

### Reasoning 훈련

훈련 데이터는 각각 15개 항목이 있는 9개 카테고리로 구성됩니다. 각 카테고리에서 14개 항목은 훈련에 사용되고 1개는 검증에 사용됩니다:

- 간단한 산술
- 간단한 if 문
- 간단한 반복문
- 반복문과 if 조합
- 간단한 상태 관리
- 재귀 함수
- 포인터 조작
- 문자열 조작
- 정렬 알고리즘

## 수식

**GRPO**(Group Relative Policy Optimization)와 **PPO**(Proximal Policy Optimization) 모두 정책 업데이트를 제한하기 위해 clipped surrogate objective를 사용하는 기본 개념을 공유합니다. 그러나 GRPO는 critic(value network)으로 계산된 토큰당 advantage를 프롬프트당 여러 샘플 완성에서 계산된 그룹 상대적 advantage로 대체하여 LLM 미세 조정에 특별히 이 접근 방식을 적용합니다.

### PPO Objective

**PPO**에서 objective는 일반적으로 다음과 같이 표현됩니다:

$$
J_{PPO}(\theta) = \mathbb{E}\Bigg[\min\Bigg(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\,A_t,\; \text{clip}\Big(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)},\,1-\epsilon,\,1+\epsilon\Big) A_t\Bigg)\Bigg],
$$

여기서:

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 는 확률 비율입니다
- $A_t $ 는 일반적으로 학습된 가치 함수를 사용하여 Generalized Advantage Estimation(GAE)을 통해 추정되는 advantage입니다
- $\epsilon$ 은 업데이트를 "proximal"하게 유지하기 위한 클리핑 매개변수입니다

이 공식은 각 타임스텝에서 $$A_t$$를 계산하기 위해 별도의 가치 네트워크(critic)를 유지해야 하는데, 보상이 희소하거나 생성된 시퀀스의 끝에서만 제공될 때 LLM에서 어려움이 될 수 있습니다.

### GRPO Objective와 그 차이점

**GRPO**(Group Relative Policy Optimization)의 핵심 아이디어는 단순한 평가기(신경망이 아님)를 사용하여 동일한 프롬프트에 대해 생성된 여러 응답에서 그룹 상대적 advantage를 계산하여 추가 가치 네트워크의 필요성을 제거하는 것입니다.

주어진 프롬프트 $$p$$에 대해, 시스템은 $$G$$ 응답 그룹을 샘플링하고 그들의 보상 $$\{r_1, r_2, \dots, r_G\}$$을 얻습니다. 각 샘플의 advantage는 다음과 같이 계산됩니다:

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}.
$$

이는 각 응답이 그룹 내 다른 응답들과 비교하여 얼마나 더 좋은지(또는 나쁜지)를 효과적으로 측정합니다.

**GRPO** objective는 다음과 같이 표현됩니다:

$$
J_{GRPO}(\theta) = \frac{1}{G}\sum_{i=1}^G \min\!\Bigg(\frac{\pi_\theta(r_i|p)}{\pi_{\theta_{old}}(r_i|p)}\,\hat{A}_i,\; \text{clip}\!\Big(\frac{\pi_\theta(r_i|p)}{\pi_{\theta_{old}}(r_i|p)},\,1-\epsilon,\,1+\epsilon\Big)\hat{A}_i\Bigg) - \beta\, D_{KL}\big(\pi_\theta \,\|\, \pi_{ref}\big).
$$

이는 정규화를 위해 그룹화된 PPO 출력에 advantage를 적용하고, catastrophic forgetting을 방지하기 위해 KL regularization을 사용합니다.

#### PPO와 비교한 주요 차이점:

**1. Advantage 추정:**
- *PPO*: 궤적을 통해 critic(value network)을 사용하여 각 토큰에서 $$A_t$$를 계산합니다
- *GRPO*: 그룹의 평균 및 표준 편차에 상대적인 보상을 정규화하여 생성된 응답당 단일 advantage를 계산합니다—특히 보상(예: 정확성 또는 서식)이 종종 시퀀스의 끝에서만 제공되는 LLM에 유용합니다

**2. 추가 가치 네트워크 없음:**
- 그룹 보상에서 직접 advantage를 도출함으로써, GRPO는 별도의 가치 함수 필요성을 제거하여 훈련 파이프라인을 단순화하고 메모리 및 계산 오버헤드를 줄입니다

**3. KL Regularization:**
- 두 방법 모두 업데이트된 정책을 참조 정책에 가깝게 유지하기 위해 KL 발산 항을 통합합니다
- GRPO에서는 이것이 보상 신호가 아닌 손실에 직접 통합되어, advantage 추정을 복잡하게 하지 않고 안정성을 유지합니다

### 수식 세부 사항

#### 보상 및 Advantage

Advantage는 다음과 같이 정의됩니다:

$$
\hat{A}_i = \frac{r_i - \mu}{\sigma + \epsilon},
$$

여기서:
- $r_i$는 $i$번째 응답에 대한 보상입니다
- $\mu$는 그룹의 평균 보상입니다
- $\sigma$는 보상의 표준 편차입니다
- $\epsilon$은 수치 안정성을 위해 추가된 작은 상수입니다

**이론적 범위:**
- 이론적으로, $r_i - \mu$는 모든 실수일 수 있고 $\sigma > 0$(보상이 모두 동일하지 않다고 가정)이므로, 정규화된 advantage $\hat{A}$는 $(-\infty, +\infty)$의 모든 값을 가질 수 있습니다

**실제 범위:**
- 실제로, 보상이 대략 정규 분포를 따를 때, $\hat{A}$는 z-점수처럼 동작합니다
- 정규 분포의 경우, 약 99.7%의 값이 $[-3, 3]$ 내에 있습니다
- 따라서, 대부분의 $\hat{A}$ 값은 일반적으로 약 $-3$에서 $3$ 사이입니다

**코드 구현:**
```python
print_logits_ids("model response", response_truncated_logits, respone_ids)
model_log_logits = selective_log_softmax(response_truncated_logits, respone_ids, tokenizer).check_shape(RESPONSE_IDS)
model_log_logits.log()

print_logits_ids("old model response", old_response_truncated_logits, respone_ids)
old_model_log_logits = selective_log_softmax(old_response_truncated_logits, respone_ids, tokenizer).check_shape(RESPONSE_IDS)
old_model_log_logits.log() 

probability_ratio = torch.exp(model_log_logits - old_model_log_logits).check_shape(RESPONSE_IDS)
probability_ratio.log() 

# 배치당 평균 및 표준편차 계산(dim=1을 따라) 및 원래 크기와 일치하도록 반복
mean_rewards = advantages.mean(dim=1).repeat_interleave(group_size).check_shape([batch_size]).check_range(0, 2.24)
mean_rewards.log()
std_rewards = advantages.std(dim=1).repeat_interleave(group_size).check_shape([batch_size]).check_range(0, float('inf'))
std_rewards.log() 

# 원래 형태로 다시 변형
advantages = advantages.view(-1)
advantages.check_shape([batch_size]).log("advantages before A_hat")
A_hat = ((advantages - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1).check_shape([batch_size, 1]) 
A_hat = torch.clamp(A_hat, -5, 5) # 작은 샘플이 너무 많은 변동을 일으키므로 클리핑 추가

# 5. grouped_ppo Loss 계산
print_step("5. Grouped ppo Loss Calc")            
# PPO objective 계산
unclipped_objective = probability_ratio
unclipped_objective.check_shape(RESPONSE_IDS).log()
epsilon_high = torch.full_like(unclipped_objective, 1 + epsilon).check_shape(RESPONSE_IDS)
_grouped_ppo_loss = - torch.minimum(unclipped_objective, epsilon_high)
_grouped_ppo_loss.check_shape(RESPONSE_IDS).log("before A_hat multiply")
_grouped_ppo_loss = _grouped_ppo_loss * A_hat
grouped_ppo_loss = _grouped_ppo_loss.mean(dim=-1).check_shape([batch_size])
grouped_ppo_loss.log() # 샘플 epsilon=0.2
```

여기서 $G$ 샘플 배치의 평균 및 분산은 다음과 같이 주어집니다:

$$
\mu = \frac{1}{G} \sum_{i=1}^{G} r_i, \quad \sigma = \frac{1}{G} \sum_{i=1}^{G} (r_i - \mu)^2.
$$

#### 확률 비율

확률 비율은 다음과 같이 정의됩니다:

$$
r_i(\theta) = \exp\left(\log \pi_\theta(r_i \mid p) - \log \pi_{\theta_{\text{old}}}(r_i \mid p)\right).
$$

**값 범위:**
- ${\pi_\theta(r_i \mid p)}\in(0,1]$
- $\log {\pi_\theta(r_i \mid p)}\in(−\infty,0]$

**지수화 후:**
$r(\theta) = \exp(\Delta \ell)$ 연산은 다음을 제공합니다:
- 범위: $\Delta \ell \in (-\infty, +\infty)$, $r(\theta) \in (0, \infty)$
- $\Delta \ell = 0$이면, $r(\theta) = \exp(0) = 1$
- $\Delta \ell > 0$(새 정책이 더 높은 확률 할당)이면, $r(\theta) > 1$
- $\Delta \ell < 0$이면, $0 < r(\theta) < 1$

이는 다음과 동등합니다:

$$
r_i(\theta) = \frac{\pi_{\theta}(r_i \mid p)}{\pi_{\theta_{\text{old}}}(r_i \mid p)}
$$

로그 공간 공식은 수학적 안정성을 위해 사용됩니다.

#### 그룹화된 PPO 손실

개별 surrogate 손실(PPO)은 다음과 같이 주어집니다:

$$
L_i(\theta) = \min\Big( r_i(\theta) \, \hat{A}_i, \, \text{clip}\big(r_i(\theta),\, 1-\epsilon_{\text{clip}},\, 1+\epsilon_{\text{clip}}\big) \, \hat{A}_i \Big).
$$

이는 클리핑이 어떻게 작동하는지에 대한 오해를 불러일으킬 수 있습니다. 실제 의도는 다음과 같습니다:

$$
L_i(\theta) =
\begin{cases}
\min\Bigl(r_i(\theta) \, A^{\pi_{\theta_t}}(s,a_i),\, (1+\epsilon) \, A^{\pi_{\theta_t}}(s,a_i)\Bigr)
& \text{if } A^{\pi_{\theta_t}}(s,a_i) > 0, \\[1ex]
\max\Bigl(r_i(\theta) \, A^{\pi_{\theta_t}}(s,a_i),\, (1-\epsilon) \, A^{\pi_{\theta_t}}(s,a_i)\Bigr)
& \text{if } A^{\pi_{\theta_t}}(s,a_i) < 0.
\end{cases}
$$

PPO 손실은 다음과 같습니다:
$$
\text{loss} = - L_i(\theta)
$$

**이론적 범위:**
- $L_i(\theta)\in(0, (1+\epsilon) A^{\pi_{\theta_t}}(s,a_i)]\text{ if } A^{\pi_{\theta_t}}(s,a_i) > 0$
- $L_i(\theta)\in(0, (1-\epsilon) A^{\pi_{\theta_t}}(s,a_i)]\text{ if } A^{\pi_{\theta_t}}(s,a_i) < 0$

(참고: 저자는 $(1-\epsilon)$ 대신 $(-1-\epsilon)$이어야 하는지, 그리고 왜 여기에 클리핑이 존재하는지 의문을 제기하며, 상한선만 더 효율적으로 보인다고 제안합니다)

**KL 발산에 대한 코드 구현:**
```python
# 3. kl_div Loss 계산
print_step("3. kl_div Loss Calc")    
# 토큰 수준 로그 확률 계산
model_log_probs = selective_log_softmax(full_shift_logits, full_shift_ids, tokenizer)
model_log_probs.log() # RESPONSE_IDS
ref_log_probs = selective_log_softmax(ref_full_shift_logits, full_shift_ids, tokenizer)
ref_log_probs.log() # RESPONSE_IDS

# 토큰 수준 KL 발산 계산
token_kl_div = F.kl_div(model_log_probs, ref_log_probs, reduction='none', log_target=True).check_shape(FULL_IDS)
token_kl_div.log()
kl_div = token_kl_div.mean(dim=-1).check_shape([batch_size])
kl_div.log() # 토큰에 대한 평균, 범위 (0, 무한대)이지만 유사한 모델의 출력의 경우 매우 작습니다. 샘플: kl_div=0.09
```

**참고 자료:**
OpenAI Spinning Up in Deep RL, Proximal Policy Optimization: [Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

전체 surrogate 손실(Grouped PPO)은 다음과 같이 계산됩니다:

$$
L_{\text{grouped\_PPO}}(\theta) = \frac{1}{G} \sum_{i=1}^{G} L_i(\theta).
$$

#### KL 페널티

KL 페널티 항은 다음과 같이 정의됩니다:

$$
L_{KL}(\theta) = \beta \, D_{KL}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big).
$$

**이론적 범위:** $L_{KL}(\theta)\in[0, \infty)$

**실제 범위:** 실제로, 정책이 유사할 때 KL 값은 종종 작습니다(예: 0.01에서 0.1).

#### 총 손실

총 손실 함수는 surrogate 손실(Grouped PPO)과 KL 페널티를 결합합니다:

$$
L_{\text{total}}(\theta) = L_{\text{grouped\_PPO}}(\theta) + L_{KL}(\theta).
$$

**코드 구현:**
```python
# kl_lambda는 KL 항에 대한 스케일링 팩터입니다
_combined_loss = grouped_ppo_loss + kl_lambda * kl_div
_combined_loss.check_shape([batch_size]).log() 
combined_loss = _combined_loss.mean()
combined_loss.log() # []
```

#### 수식 조정

저자는 두 가지 주요 수정 사항을 언급합니다:
1. 작은 샘플이 너무 많은 변동을 일으키므로 클리핑을 추가했습니다
2. objective를 클리핑하는 대신, 더 나은 결과를 생성하는 상한을 사용합니다

## 기술적 어려움

프로젝트는 다음과 같은 여러 과제에 직면했습니다:

1. **CUDA 문제**: AMD 제품보다 낫지만, CUDA에는 많은 버그가 있어 이전 드라이버 버전으로 롤백하고 전체 GPU 기반 실행이 필요했습니다

2. **ChatGPT 불일치**: 도움이 되지만, ChatGPT는 종종 버그가 있는 코드를 제공했습니다. 예시:
   - ID와 logit 사이의 불필요한 정렬
   - 텐서 차원에 대한 오해의 소지가 있는 코멘트가 있는 부정확한 코드 조각
   - 함수 업데이트 시 중요한 매개변수 제거
   - API 문서와 실제 동작 사이의 불일치

3. **제한된 GPU 메모리**: 간단한 LLM 훈련에도 49GB VRAM이 충분하지 않아, 효율적인 코드에서 라인별 메모리 최적화가 필요했습니다. 처음에는 7B 모델을 훈련하려 했지만, 부분 레이어 훈련으로 3B 모델로 타협해야 했습니다. 최적화 프로그램과 현재 LLM 모델의 불규칙한 구조는 예상보다 훨씬 더 많은 메모리를 요구합니다.

4. **버그가 있는 수식**: 잘 알려진 코드 구현에도 종종 수식 버그가 포함되어 있습니다. 상당한 컴퓨팅 파워와 대규모 샘플 크기로는 이러한 버그가 최종 결과에 영향을 미치지 않을 수 있지만, 제한된 리소스와 낮은 정밀도에서는 작은 실수가 결과에 크게 영향을 미칩니다.

5. **시간 제약**: 한 에포크가 너무 오래 걸리고, 구성할 하이퍼파라미터가 너무 많습니다. Optuna와 같은 도구가 일부 하이퍼파라미터 범위를 제거할 수 있지만, 과정은 여전히 시간이 많이 소요됩니다.

## 배운 교훈

1. **수식을 완전히 이해하기**: 수식이 훈련에 왜, 어떻게 적용되어야 하는지 이해하세요
2. **ChatGPT를 믿지 말고 검증하세요**: 종종 가짜 증거로 답변을 제공합니다. 코드를 실행하고, 결과를 기록하고, 독립적으로 검증하세요
3. **수작업으로 만든 데이터 사용하기**: 맞춤형 데이터는 종종 기존의 데이터셋보다 더 잘 작동합니다
4. **올바른 프롬프트 형식 찾기**: 모델이 훈련된 형식은 잘 알려진 형식보다도 훨씬 더 잘 작동하는 경우가 많습니다

## 도움이 되는 자료

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [Coding GRPO from Scratch: A Guide to Distributed Implementation with QWen2.5–1.5B-Instruct](https://blog.gopenai.com/coding-grpo-from-scratch-a-guide-to-distributed-implementation-with-qwen2-5-1-5b-instruct-59b34227edac)
- [Introducing the Clipped Surrogate Objective Function](https://huggingface.co/learn/deep-rl-course/en/unit8/clipped-surrogate-objective)

개념을 이해하는 것과 구현하는 것이 매우 다른 경험이었습니다다. 이론적으로 명확해 보이는 것이 구현 과정에서 이해의 간극을 드러냈습니다. "Coding GRPO from Scratch"는 구현 과제에 직면했을 때 특히 도움이 되었습니다.

## 부록: 모델 실행 방법

### 시퀀스 생성 및 보상 계산

```python
# 1. 생성을 위한 모델 forward pass
print_step("1. Model train")
with torch.no_grad():
    full_ids, truncated_ids, respone_ids, prompt_lengths = generate_ids(model, batch, tokenizer, temperature)
    full_text_lists = tokenizer.batch_decode(truncated_ids, skip_special_tokens=True)
    reward_work.reward(full_text_lists, writer, log_group, global_step)
    # 생성에서 사용되지 않는 텐서 해제
    full_text_lists = None
```

### 훈련 및 Logits 획득을 위한 모델 실행

```python
_full_shift_logits, response_truncated_logits, _ = compute_logits(model, full_ids, prompt_lengths, respone_ids, tokenizer) 
```

### 이전 모델과 참조 모델 출력 가져오기

이전 모델은 reasoning 훈련에 사용되고, 참조 모델은 catastrophic forgetting을 방지합니다:

```python
# 2. 레거시 모델 실행(이전 및 참조 모델)
print_step("2. Legacy Models Run")
with torch.no_grad():
    _, old_response_truncated_logits, _ = compute_logits(old_model, truncated_ids, prompt_lengths, respone_ids, tokenizer, detach_out=True)
    ref_full_shift_logits, _, _ = compute_logits(ref_model, full_ids, prompt_lengths, respone_ids, tokenizer, detach_out=True)
```

### 훈련 없이 시퀀스 생성

```python
def generate_ids(model, batch, tokenizer, temperature):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    eos_token_id = tokenizer.eos_token_id

    # 배치의 각 예제에 대한 프롬프트 길이를 EOS의 첫 번째 발생에 기반하여 결정
    prompt_lengths = []
    for i in range(input_ids.size(0)):
        seq = input_ids[i]
        # 토큰이 eos_token_id와 같은 인덱스 찾기
        eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        # 최소 하나의 발생이 있으면, 그 인덱스 + 1 사용(프롬프트에 EOS를 포함하려는 경우)
        # 그렇지 않으면, 전체 시퀀스 길이로 폴백
        if eos_positions.numel() > 0:
            first_eos = eos_positions[0].item() + 1
        else:
            first_eos = seq.size(0)
        prompt_lengths.append(first_eos)
    
    print_memory("Prompt lengths per batch element: " + str(prompt_lengths))

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,  # max_length가 전역적으로 정의되어 있다고 가정
        temperature=temperature,
        do_sample=True,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        output_scores=True,
        return_dict_in_generate=True
    )
    full_ids = output.sequences.detach()
    truncated_ids = cut_ids_on_eos_tensor(full_ids, tokenizer.eos_token_id)    
    respone_ids = pad_sequence([truncated_ids[idx][p_len:] for idx, p_len in enumerate(prompt_lengths)],
                                batch_first=True, padding_value=tokenizer.pad_token_id)
    truncated_ids = pad_sequence(truncated_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    full_ids = pad_sequence(full_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    output = None
    print_memory("full_ids.shape[-1]: " + str(full_ids.shape[-1]))
    return full_ids, truncated_ids, respone_ids, prompt_lengths
```

### 모델 훈련을 위한 Logits 생성

```python
def compute_logits(model, full_ids, prompt_lengths, respone_ids, tokenizer, detach_out=False):
    # full_ids 리스트를 (batch, max_seq_length) 형태의 전체 텐서로 패딩
    full_ids = pad_sequence(full_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(dtype=torch.int32)
    
    # 비패딩 토큰은 1, 패딩 토큰은 0인 attention mask 생성
    full_ids_mask = (full_ids != tokenizer.pad_token_id).to(dtype=torch.int32, device=full_ids.device)
    
    # 전체 패딩된 텐서에 대한 logits 계산
    logits = model(input_ids=full_ids, attention_mask=full_ids_mask, early_stop=False).logits
    
    truncated_response_ids_list = []
    truncated_response_logits_list = []
    batch_size = full_ids.size(0)
    
    for i in range(batch_size):
        p_len = prompt_lengths[i]
        # 이 배치 요소에 대한 실제 시퀀스 길이(패딩 무시) 결정
        actual_length = full_ids_mask[i].sum().item()
        # 프롬프트 길이가 실제 길이를 초과하지 않도록 함
        if p_len > actual_length:
            p_len = actual_length

        # 이 예제에 대한 완성 토큰 ID 추출
        comp_ids = full_ids[i, p_len:actual_length].detach()
        # logits의 경우, 완성 바로 전의 토큰을 포함하려면 p_len-1부터 슬라이스
        comp_logits = logits[i, p_len-1:actual_length-1, :]
        
        # 예상 응답 길이에 맞게 길이 조정
        expected_len = respone_ids.shape[1]
        truncated_response_ids_list.append((comp_ids.detach() if detach_out else comp_ids)[:expected_len])
        truncated_response_logits_list.append((comp_logits.detach() if detach_out else comp_logits)[:expected_len, :])

    truncated_response_logits = pad_sequence(truncated_response_logits_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    truncated_response_ids = pad_sequence(truncated_response_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    return logits, truncated_response_logits, truncated_response_ids
```