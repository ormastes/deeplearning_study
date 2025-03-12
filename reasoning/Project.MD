# Project 요약
이름: 윤종현

## 하는 일
SSD FW 개발. C/C++ 기반 osless baremetal FW 개발.
취미로 Typescript/Python 기반 개발 Tool 개발

## Project subject
VScode Test extension 에 대한 Test code 자동 생성.

| Test Target                             | Target Object     | Input Data                 | Expected Output | Clang-repl Test                                                             |
| --------------------------------------- | ----------------- | -------------------------- | --------------- | --------------------------------------------------------------------------- |
| int add(int a, int b)<br/>{return a+b;} | Add two variables | int a = 1;</br> int b = 2; | %<< result 3    | int a = 1;</br>int b = 2;</br>int result = add(a, b);</br>%<< result;</br>3 |
|                                         |                   |                            |                 |                                                                             |

#### Training Data
##### 기본 Data
1. CPP-UT-Bench:
   https://huggingface.co/datasets/Nutanix/CPP-UNITTEST-BENCH
2. CompCodeVet: Compiler-Validated Data Curation
   https://huggingface.co/datasets/Elfsong/Mercury
   https://github.com/Elfsong/Mercury/blob/main/src/dpo_train.py
   https://github.com/Elfsong/Mercury/blob/main/src/sft_train.py
3. <strike>CITYWALK: Enhancing LLM-Based C++ Unit Test Generation
   https://zenodo.org/records/14022506 </strike><< not a dataset.
   But the logic can be used for sample data, and training data refine.

상기 데이터를 Refine하여 사용

## 예상되는 어려운 점
1. Fine tuning 시 학습이 재대로 되지 않고 Model이 망가지는 경우.
   * 일부 Layer만 학습으로 방지
2. <strike>Test Data를 구하거나 생성하는 것이 어려울 수 있음.
   * Test Data 확인 완료.</strike> (완료)
3. <strike>GPU Resource 가 모자를 수 있음.
   * WizardCoder (https://ollama.com/library/wizardcoder:7b-python) 선택
     * Deepseek coder 대신 WizardCoder 를 선택한 이유는 WizardCoder는 MoE를 적용하지 않아 Study에 적합하다.
   * 7B model minimum: 28GB
   * with AdamW: 56GB
   * with AdaFactor: 20GB~28GB
   * AdaFactor와 몇몇 Layer만 선택적으로 Training</strike> (완료)

## Training Plan
### 1. sample data exact matching training
About 1000 sample data will be used for training.

### 2. Check tag and result training
check Tag ("Target Object", "Input Data", "Expected Output") are exists.
Result ("Clang-repl Test") is correct.
when the result is not correct, check there is error.

## Project 목표
### 기본 목표
1. "C" 언어 기반 clang-repl Test Case 생성

### 추가 목표
1. "C++" 언어 기반 clang-repl Test Case 생성
2. Vscode extension 을 통한 Test Case 생성
3. Clangd LSP 를 통한 Source Code RAG 적용 (https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/)
4. Clang-repl interpreter Agent 적용
   - 환경 setting load
   - clang-repl 실행 Training add (Practice step add)
5. Mixture of Expert (MoE) 적용

## 학습 목표
1. Reasoning 학습 (기본 목표: "C" 언어 기반 clang-repl Test Case 생성)
   - Clang-repl interpreter Agent 적용
2. RAG 학습 (추가 목표: Clangd LSP 를 통한 Source Code RAG 적용)
3. MoE 학습 (추가 목표: Mixture of Expert (MoE) 적용)

## Action Items
1. <strike>Acquire Training Data</strike> V
2. <strike>Acquire Model</strike> V
3. <strike>Analyze Training Data</strike> V
4. Simple Training with AdaFactor and WizardCoder
   * Log, tensorboard, and model save template.
   * Simple prompt embedding training
5. Refine Training Data to form reasoning sample
   * Mercury data to train Target Object Extraction
   * CPP-UNITTEST-BENCH to train Clang-repl Test Case complete sample
6. Training with reasoning sample (1. sample data exact matching training)
   * Mercury data to train Target Object Extraction
   * CPP-UNITTEST-BENCH to train Clang-repl Test Case complete sample

## Additional Action Items
1. Training with reasoning (2. Check tag and result training)
   * Mercury data to Tag and Result training
2. VSCode extension development
   * Menu on function
     - Take function name with location
     - Whole file
     - Class declaration location if exists
   * Check existing test case
3. Interface with GitHub (url, branch)
   * automated build check
   * List sample function from loaded source code
4. Interface with Clangd LSP (RAG)
5. Interface with Clang-repl > cdoctest python interface, takes all symbols in function and its definition.
   * high level reasoning (Practice step add)
6. Apply Mixture of Expert (MoE)`

## Reference
1. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
 Reinforcement Learning: https://arxiv.org/pdf/2501.12948
2. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models: https://arxiv.org/abs/2402.03300
3. CPP-UT-Bench: Can LLMs Write Complex Unit Tests in C++?: https://arxiv.org/abs/2412.02735
4. CompCodeVet: A Compiler-guided Validation and Enhancement Approach for Code Dataset: https://arxiv.org/abs/2311.06505
5. CITYWALK: Enhancing LLM-Based C++ Unit Test Generation: https://arxiv.org/abs/2501.16155
6. Coding GRPO from Scratch: A Guide to Distributed Implementation with QWen2.5–1.5B-Instruct: https://blog.gopenai.com/coding-grpo-from-scratch-a-guide-to-distributed-implementation-with-qwen2-5-1-5b-instruct-59b34227edac
7. Simple Reinforcement Learning for Reasoning: https://github.com/hkust-nlp/simpleRL-reason
7. RAG From Scratch: https://github.com/langchain-ai/rag-from-scratch
8. Towards Understanding the Mixture-of-Experts Layer in Deep Learning: https://par.nsf.gov/servlets/purl/10379033 , https://github.com/uclaml/MoE?tab=readme-ov-file
9. Mixture of Experts from scratch(antonio-f): https://github.com/antonio-f/mixture-of-experts-from-scratch/blob/main/moe.ipynb
10. Expert Parallelism Load Balancer (EPLB): https://github.com/deepseek-ai/eplb
11. Fire-Flyer File System: https://github.com/deepseek-ai/3FS

## 향후
### 추가 Data
cmake/clang buildable c++ code.
> llvm : https://github.com/llvm/llvm-project

### 향후 목표
1. C to bitcode 변환
2. bitcode to C 변환
3. Python to bitcode 변환
4. bitcode to Python 변환
5. Python to C 변환


### Future Action Items
1. Interface to build c to bitcode
2. Interface to build python(Numba) to bitcode
3. Prepare bitcode data for C to bitcode conversion
4. Prepare bitcode data for Python to bitcode conversion
5. Training with bitcode data
6. Problem-solving training with pseudocode(python)
   * Interface with python-repl

### Future Action Items (TDD)
1. TDD applied training.
   * Make complete sample which modify Target Code but still pass the test.
   * Train to generate what if target code generation.
   * Train to add test case for what if target code. to fail.
   * Repeat until no more test case can be added. Notify "No more test/Trying add test case"
   * Train stop signal base on coverage.
2. Target Code Logic training.
   * make a sample one solution of Mercury can pass but other solution can't pass.
   * not perfect solution with other tests are given and "No more test" notified.
   * Train to generated target code to fail.
     * + if generate failing tc
     * - if generate error/success tc
     * - if it notifies done.
