# 1) Re-instantiate base TinyLlama
from llm.tinyllama.core.config import Config

cfg = Config.from_json("path/to/pretrained_1.1b/config.json")
base_model = TinyLlama(cfg)
base_model.load_state_dict(torch.load("path/to/pretrained_1.1b/pytorch_model.bin"))

# 2) Wrap in the same LoRA configuration
from peft import get_peft_model

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model_with_lora = get_peft_model(base_model, lora_config)

# 3) Load LoRA weights
model_with_lora.load_state_dict(torch.load("tinyllama_lora_step30000.pt"), strict=False)
model_with_lora.eval()
model_with_lora.to(device)
