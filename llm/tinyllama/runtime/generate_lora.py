# Get the directory where this file (loader.py) is located
import os
import time

import torch
from peft import get_peft_model, LoraConfig, TaskType

from llm.tinyllama.core.model import TinyLlama
from llm.tinyllama.io.hf_state import load_state, load_state_pt
from llm.tinyllama.io.tokenizer import Tok
from llm.tinyllama.qa.q_a import get_answer




current_dir = os.path.dirname(os.path.abspath(__file__))

# Set model directory relative to this file's location
model_dir = os.path.join(current_dir, "../pretrained_1.1b")

# Determine device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer with absolute path
tokenizer_path = os.path.join(model_dir, "tokenizer.model")
tokenizer = Tok(tokenizer_path)


# Load model config with absolute path
config_path = os.path.join(model_dir, "config.json")
model = TinyLlama.from_config(config_path).to(device)

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

pt_path = os.path.join(model_dir, "tinyllama_lora_step7000.pt")
# Load state dict
with torch.no_grad():
    missing, unexpected = model.load_state_dict(load_state_pt(pt_path))
    if missing:
        print(f"Missing keys in state dict: {missing}")
    if unexpected:
        print(f"Unexpected keys in state dict: {unexpected}")


model.eval() # Ensure model is in evaluation mode

pairs = [
    {
        "system": "You are a pentest agent. Given the following state and numbered actions, choose the correct next action.",
        "user": '''You should response with the json format, contains the 'reason' and 'best_action' fields. The best_action contains 'index' and 'module_name' fields.
Current state: a Windows host () has been identified with vulnerabilities CVE-2024-29824.
Available actions:
1. {'module_path': 'multi/http/trendmicro_threat_discovery_admin_sys_time_cmdi', 'platforms_supported': ['Linux'], 'type': 'remote_http_exploit', 'prerequisites': [], 'associated_vuln': ['CVE-2016-7552', 'CVE-2016-7547']}
2. {'module_path': 'windows/http/ivanti_epm_recordgoodapp_sqli_rce', 'platforms_supported': ['Windows'], 'type': 'remote_http_exploit', 'prerequisites': [], 'associated_vuln': ['CVE-2024-29824']}
3. {'module_path': 'windows/smtp/mailcarrier_smtp_ehlo', 'platforms_supported': 'Windows, Windows 2000 SP0 - XP SP1 - EN/FR/GR, Windows XP SP2 - EN', 'type': 'generic_exploit', 'prerequisites': [], 'associated_vuln': ['CVE-2004-1638', 'OSVDB-11174', 'BID-11535', 'EDB-598']}
Which action should be executed next?
'''
    },
    # {
    #     "system": "You are an expert in algorithms and data structures.",
    #     "user": "Answer in short. Explain how Dijkstra’s algorithm works, step by step, on a small weighted graph."
    # },
    # {
    #     "system": "You are a creative writing assistant.",
    #     "user": "Answer in short. Write a 200-word opening scene for a sci-fi story set on Mars."
    # },
    # {
    #     "system": "You are a financial analyst.",
    #     "user": "Answer in short. Compare the historical performance of the S&P 500 and NASDAQ over the last decade."
    # },
    # {
    #     "system": "You are a cooking tutor.",
    #     "user": "Answer in short. How do I make a classic French omelette with a soft, custardy interior?"
    # }
]

durations_no_cache = []
durations_with_cache = []

for pair in pairs:
    sys_msg_input = pair["system"]
    usr_msg_input = pair["user"]

    if not sys_msg_input.strip() or not usr_msg_input.strip():
        print("System message and user question cannot be empty. Please try again.")
        continue

    # start_time = time.time()
    # generated_answer, confidence = get_answer(sys_msg_input, usr_msg_input, model, tok, 1024, no_cache=True)
    # end_time = time.time()
    #
    # print(f"\nAssistant: {generated_answer}")
    # print(f"(Confidence: {confidence:.2f}, Time: {end_time - start_time:.2f}s)")

    # durations_no_cache.append(end_time - start_time)

    start_time = time.time()
    generated_answer, confidence = get_answer(sys_msg_input, usr_msg_input, model, tokenizer, 1024)
    end_time = time.time()

    print(f"\nAssistant: {generated_answer}")
    print(f"(Confidence: {confidence:.2f}, Time: {end_time - start_time:.2f}s)")

    durations_with_cache.append(end_time - start_time)