import torch
from peft import LoraConfig, TaskType, get_peft_model

from llm.tinyllama import loader

tokenizer, model = loader.initialize_model()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # because TinyLlama is a causal language model
    inference_mode=False,           # we intend to train (not just inference)
    r=8,                             # LoRA rank (4–16 is a common range; start with 8)
    lora_alpha=32,                   # LoRA α (scaling)
    lora_dropout=0.05,               # dropout on the LoRA layers
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
)

# 3) Wrap TinyLlama
model = get_peft_model(model, lora_config)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 4) Print total vs. LoRA parameter counts
def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

total = sum(p.numel() for p in model.parameters())
trainable = count_trainable_params(model)
print(f"Total parameters: {total/1e6:.1f}M")
print(f"Trainable parameters (LoRA only): {trainable/1e6:.1f}M")