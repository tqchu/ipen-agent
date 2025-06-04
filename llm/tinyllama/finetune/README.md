### 1. Setup

Load pretrained TinyLlama-1.1 B.

Insert your chosen adapters/LoRA modules (if using).

Freeze whatever you want (e.g., embeddings + early layers).

Use a language modeling objective where, for each instruction pair, the model sees:

`<s> {instruction} {input} <eos> {output} <eos>`

Compute cross-entropy loss only on the tokens of {output}, not on {instruction}.

### 2. Hyperparameters (example starting point)

Batch size: 32 sequences (adjust for your GPU).

Learning rate: 1e-4 (for adapters/LoRA) or 5e-5 (for full fine-tuning).

Warmup steps: 500–1000.

Training steps: 10k–50k (monitor val‐loss).

Weight decay: 0.01 (optional, to prevent overfitting).

Gradient clipping: 1.0.

Loss Masking

Mask out all tokens in the “instruction” portion so that the gradient flows only through the “output” portion.

If you have chain-of-thought labels but you don’t want to reveal them at inference, you can mask them too (i.e., only compute loss on the final “answer”).

Validation & Checkpointing

Every 500–1000 steps, evaluate on the val set. If val-loss plateaus or starts to rise, either lower LR or stop.

Save a checkpoint whenever val-loss improves.

### 3. Prompt Engineering & Instruction Tuning
Prompt Templates

Standardize how you feed questions:

`“You are a professional pentester. {instruction} 
 Provide step-by-step commands, expected outputs, and security considerations.”
Including “You are a professional pentester” helps the model stay on domain.`

Chain-of-Thought Prompts

If you want your model to “show its reasoning,” prefix your examples with “Let me think step by step.”

During inference, you can also ask “Explain your reasoning in detail.”

### 4. Evaluation
Exact-Match vs. F1

For certain constrained prompts (e.g., “Which Nmap script should I run?”) you can evaluate exact command match.

For broader “Describe how to exploit CVE-2021-44228,” you can measure semantic overlap (e.g., BERTScore or GPT-based RAG metrics).

Red Team / Blue Team Cycles

Have a second team attempt to “use” the model’s instructions on a lab environment (e.g., a vulnerable VM). See if the step sequence actually works.

This gives you ground truth of “did the model’s output successfully compromise the target?” which is the ultimate metric.

Human-In-The-Loop Testing

Give your pentesters blind tests: half the prompts answered by your fine-tuned TinyLlama, half answered by a baseline (e.g., GPT-3.5). See if they prefer one or the other under timed conditions.