# Project 요약
이름: 윤종현

## 하는 일
SSD FW 개발

## Project subject
VScode Test extension 에 대한 Test code 자동 생성.

| Test Target                             | Target Object     | Input Data                 | Expected Output | Clang-repl Test                                                             |
| --------------------------------------- | ----------------- | -------------------------- | --------------- | --------------------------------------------------------------------------- |
| int add(int a, int b)<br/>{return a+b;} | Add two variables | int a = 1;</br> int b = 2; | %<< result 3    | int a = 1;</br>int b = 2;</br>int result = add(a, b);</br>%<< result;</br>3 |
|                                         |                   |                            |                 |                                                                             |

## Training Plan
### 1. sample data exact matching training
About 1000 sample data will be used for training. 
### 2. Check tag and result training
check Tag ("Target Object", "Input Data", "Expected Output") are exists.
Result ("Clang-repl Test") is correct.
when the result is not correct, check there is error.

## 목표
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

### 향후 목표
1. C to bitcode 변환
2. bitcode to C 변환
3. Python to bitcode 변환
4. bitcode to Python 변환
5. Python to C 변환

## 학습 목표
1. Reasoning 학습 (기본 목표: "C" 언어 기반 clang-repl Test Case 생성) 
   - Clang-repl interpreter Agent 적용
2. RAG 학습 (추가 목표: Clangd LSP 를 통한 Source Code RAG 적용)
3. MoE 학습 (추가 목표: Mixture of Expert (MoE) 적용)

## 학습 방법
### Base Model 선택
WizardCoder (https://ollama.com/library/wizardcoder:7b-python)
Deekseek coder 대신 WizardCoder 를 선택한 이유는 WizardCoder는 MoE를 적용하지 않아 Study에 적합하다.

### 최소 환경
#### Training Memory
7B model minimum: 28GB
with AdamW: 56GB
with AdaFactor: 20GB~28GB

AdaFactor와 몇몇 Layer만 선택적으로 Training

#### Training Data
##### 기본 Data
1. CPP-UT-Bench: 
https://huggingface.co/datasets/Nutanix/CPP-UNITTEST-BENCH
2. CompCodeVet: Compiler-Validated Data Curation
https://huggingface.co/datasets/Elfsong/Mercury
https://github.com/Elfsong/Mercury/blob/main/src/dpo_train.py
https://github.com/Elfsong/Mercury/blob/main/src/sft_train.py
3. CITYWALK: Enhancing LLM-Based C++ Unit Test Generation
https://zenodo.org/records/14022506

상기 데이터를 Refine하여 사용

##### 추가 Data
cmake/clang buildable c++ code.

## Action Items
1. Acquire Training Data
2. Acquire Model 
3. Simple Training with AdaFactor and WizardCoder
4. Refine Training Data to form reasoning sample 
## Additional Action Items
1. Training with reasoning sample (1. sample data exact matching training)
2. Training with reasoning (2. Check tag and result training)
3. VSCode extension development
4. Interface with Clangd LSP (RAG)
5. Interface with Clang-repl
    * high level reasoning (Practice step add)
6. Interface with GitHub (url, branch)
   * automated build check
   * List sample function from loaded source code
7. Apply Mixture of Expert (MoE)
### Future Action Items
1. Interface to build c to bitcode
2. Interface to build python(Numba) to bitcode
3. Prepare bitcode data for C to bitcode conversion
4. Prepare bitcode data for Python to bitcode conversion
5. Training with bitcode data
6. Problem solving training with pseudo code(python)
   * Interface with python-repl