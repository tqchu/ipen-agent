import argparse
import json
import math
import os
import time

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from llm.tinyllama import loader
from llm.tinyllama.finetune.dataset import QADataset
from llm.tinyllama.finetune.loader import collate_fn

# If you want to experiment with allocator settings, you can set this env var:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load and move model to GPU, then turn on gradient checkpointing
tokenizer, model = loader.initialize_model()
hardware = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # because TinyLlama is a causal language model
    inference_mode=False,  # we intend to train (not just inference)
    r=8,  # LoRA rank (4–16 is a common range; start with 8)
    lora_alpha=32,  # LoRA α (scaling)
    lora_dropout=0.05,  # dropout on the LoRA layers
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
)

# 3) Wrap TinyLlama
model = get_peft_model(model, lora_config)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# model.gradient_checkpointing_enable()

# 2) Create optimizer and AMP scaler
optimizer = AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()

# 3) Hyperparameters: smaller microbatch + accumulation
num_epochs = 4
per_device_batch_size = 2  # drop from 16 → 2
gradient_accumulation_steps = 4  # 2×4 = effective batch 8
save_every = 1000  # save/validate every N steps

current_dir = os.path.dirname(os.path.abspath(__file__))

train_data_file = "train_data.jsonl"
val_data_file = "val_data.jsonl"

parser = argparse.ArgumentParser(description="Train a TinyLlama model with LoRA")
parser.add_argument('--minimal', action='store_true', help='Use minimal dataset for training')
args = parser.parse_args()

print("Use minimal dataset:", args.minimal)

if args.minimal:
    train_data_file = "train_minimal_data.jsonl"
    val_data_file = "val_minimal_data.jsonl"

train_dataset = QADataset(os.path.join(current_dir, "train_data.jsonl"), tokenizer, max_length=512)
val_dataset = QADataset(os.path.join(current_dir, "val_data.jsonl"), tokenizer, max_length=512)

train_loader = DataLoader(train_dataset, batch_size=per_device_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=per_device_batch_size, shuffle=False)

vocab_size = model.config.vocab_size
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

global_step = 0
model.train()
start_time = time.time()
train_log = []  # list of (global_step, train_loss, train_ppl, train_acc)
val_log = []  # list of (global_step, val_loss,   val_ppl,   val_acc)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    step_loss_accum = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)  # (B, T)
        labels = batch["labels"].to(device)  # (B, T)

        # ----- Forward + loss under autocast (FP16) -----
        with autocast():
            logits, _ = model(input_ids, kv_caches=None)  # (B, T, V)
            shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
            shift_labels = labels[:, 1:].contiguous()  # (B, T-1)

            loss = loss_fn(
                shift_logits.view(-1, vocab_size),  # (B*(T-1), V)
                shift_labels.view(-1)  # (B*(T-1),)
            )
            loss = loss / gradient_accumulation_steps

        # ----- Backward with GradScaler -----
        scaler.scale(loss).backward()
        step_loss_accum += loss.item()
        global_step += 1

        # ----- Gradient accumulation & step -----
        if global_step % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # (optional) clip gradients here, e.g.:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += step_loss_accum
            step_loss_accum = 0.0

            # **Compute train metrics** (loss, ppl, acc)
            avg_train_loss = epoch_loss / (global_step // gradient_accumulation_steps or 1)
            # Added: compute perplexity
            avg_train_ppl = math.exp(avg_train_loss)
            # Added: compute token‐level accuracy
            with torch.no_grad():
                preds = shift_logits.argmax(-1)
                mask = shift_labels != -100
                train_acc = (preds[mask] == shift_labels[mask]).float().mean().item()

            # **Log training metrics**
            # Updated: now storing ppl and acc as well
            train_log.append((global_step, avg_train_loss, avg_train_ppl, train_acc))

        # ----- Update progress bar -----
        avg_loss = epoch_loss / (global_step // gradient_accumulation_steps or 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

        # ----- Periodic validation & checkpoint saving -----
        if global_step % save_every == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            val_acc_num = 0  # Added: counters for accuracy
            val_tok_num = 0  # Added

            with torch.no_grad():
                for vbatch in val_loader:
                    vid = vbatch["input_ids"].to(device)
                    lbl = vbatch["labels"].to(device)

                    with autocast():  # validation can also use FP16
                        v_logits, _ = model(vid, kv_caches=None)  # (B, T, V)
                        v_shift_logits = v_logits[:, :-1, :].contiguous()
                        v_shift_labels = lbl[:, 1:].contiguous()

                        v_loss = loss_fn(
                            v_shift_logits.view(-1, vocab_size),
                            v_shift_labels.view(-1)
                        )
                    val_loss += v_loss.item()
                    val_steps += 1

                    # Added: accumulate for token‐level accuracy
                    preds = v_shift_logits.argmax(-1)
                    mask = v_shift_labels != -100
                    val_acc_num += (preds[mask] == v_shift_labels[mask]).sum().item()
                    val_tok_num += mask.sum().item()

            avg_val_loss = val_loss / (val_steps or 1)
            # Added: compute perplexity & accuracy
            avg_val_ppl = math.exp(avg_val_loss)
            avg_val_acc = val_acc_num / val_tok_num if val_tok_num > 0 else 0.0

            # Updated: now storing ppl and acc as well
            val_log.append((global_step, avg_val_loss, avg_val_ppl, avg_val_acc))
            print(
                f"\n→ Step {global_step}: validation loss = {avg_val_loss:.4f}, ppl = {avg_val_ppl:.2f}, acc = {avg_val_acc:.4f}\n")
            model.train()

            # Save only the LoRA‐adapter (you can still call `model.state_dict()`
            # because LoRA layers are the only trainable weights, or use a PEFT helper)
            ckpt_path = f"tinyllama_lora_step{global_step}.pt"
            torch.save(model.state_dict(), ckpt_path)

    # end of epoch
print("Training complete.")
end_time = time.time()
total_time_s = end_time - start_time
# === Export metrics to CSV for your report ===
# Updated: include ppl and acc columns
df_train = pd.DataFrame(
    train_log,
    columns=["step", "train_loss", "train_ppl", "train_acc"]
)
df_val = pd.DataFrame(
    val_log,
    columns=["step", "val_loss", "val_ppl", "val_acc"]
)
df = pd.merge_asof(df_train, df_val, on="step", direction="backward")

df.to_csv("metrics_full.csv", index=False)

# Updated summary statistics to include ppl and acc
summary = {
    "final_train_loss": df_train["train_loss"].iloc[-1],
    "final_train_ppl": df_train["train_ppl"].iloc[-1],
    "final_train_acc": df_train["train_acc"].iloc[-1],
    "final_val_loss": df_val["val_loss"].iloc[-1],
    "final_val_ppl": df_val["val_ppl"].iloc[-1],
    "final_val_acc": df_val["val_acc"].iloc[-1],
    "best_val_loss": df_val["val_loss"].min(),
    "best_val_ppl": df_val.loc[df_val["val_loss"].idxmin(), "val_ppl"],
    "best_val_acc": df_val.loc[df_val["val_loss"].idxmin(), "val_acc"],
    "steps_to_best": int(df_val.loc[df_val["val_loss"].idxmin(), "step"]),
    "hardware": hardware,
    "total_training_time_s": total_time_s,
    "gpu_hours": total_time_s / 3600.0
}
with open("metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("→ metrics_full.csv and metrics_summary.json written.")
