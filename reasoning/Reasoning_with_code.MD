# Reasoning with code

## Lesson learned.
1. Model's actual size is much bigger than vanilla model's size base on parameters.

## Memorize Knowledge on LLM
https://docs.llamaindex.ai/en/v0.9.48/examples/finetuning/knowledge/finetune_knowledge.html
### Key point.
Make knowledge in to QnA pair dataset to train it.
### How prevent catastrophic forgetting
1. Mixed Training Data / Replay Buffer:
By including a portion of general or original QA pairs alongside your new, domain-specific Q&A dataset.
2. Regularization Techniques:
Methods such as elastic weight consolidation (EWC) or similar regularization strategies can constrain changes to weights that are crucial for maintaining the model’s base knowledge,
3. Adapter-Based Methods (e.g., LoRA):
Techniques like LoRA update only a small subset of the model’s parameters while keeping the majority of the original weights frozen. 









