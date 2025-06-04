import os
import torch
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
model.to(device)
# model.gradient_checkpointing_enable()

# 2) Create optimizer and AMP scaler
optimizer = AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()

# 3) Hyperparameters: smaller microbatch + accumulation
num_epochs = 3
per_device_batch_size = 2           # drop from 16 → 2
gradient_accumulation_steps = 4     # 2×4 = effective batch 8
save_every = 1000  # save/validate every N steps

current_dir = os.path.dirname(os.path.abspath(__file__))
train_dataset = QADataset(os.path.join(current_dir, "train_data.jsonl"), tokenizer, max_length=512)
val_dataset   = QADataset(os.path.join(current_dir, "val_data.jsonl"),   tokenizer, max_length=512)

train_loader = DataLoader(train_dataset, batch_size=per_device_batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=per_device_batch_size, shuffle=False)

vocab_size = model.cfg.vocab_size
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

global_step = 0
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    step_loss_accum = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)  # (B, T)
        labels    = batch["labels"].to(device)     # (B, T)

        # ----- Forward + loss under autocast (FP16) -----
        with autocast():
            logits, _ = model(input_ids, kv_caches=None)    # (B, T, V)
            shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
            shift_labels = labels[:, 1:].contiguous()       # (B, T-1)

            loss = loss_fn(
                shift_logits.view(-1, vocab_size),   # (B*(T-1), V)
                shift_labels.view(-1)                # (B*(T-1),)
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

        # ----- Update progress bar -----
        avg_loss = epoch_loss / (global_step // gradient_accumulation_steps or 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

        # ----- Periodic validation & checkpoint saving -----
        if global_step % save_every == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0

            with torch.no_grad():
                for vbatch in val_loader:
                    vid = vbatch["input_ids"].to(device)
                    lbl = vbatch["labels"].to(device)

                    with autocast():  # validation can also use FP16
                        v_logits, _ = model(vid, kv_caches=None)     # (B, T, V)
                        v_shift_logits = v_logits[:, :-1, :].contiguous()
                        v_shift_labels = lbl[:, 1:].contiguous()

                        v_loss = loss_fn(
                            v_shift_logits.view(-1, vocab_size),
                            v_shift_labels.view(-1)
                        )
                    val_loss += v_loss.item()
                    val_steps += 1

            avg_val_loss = val_loss / (val_steps or 1)
            print(f"\n→ Step {global_step}: validation loss = {avg_val_loss:.4f}\n")
            model.train()

            # Save only the LoRA‐adapter (you can still call `model.state_dict()`
            # because LoRA layers are the only trainable weights, or use a PEFT helper)
            ckpt_path = f"tinyllama_lora_step{global_step}.pt"
            torch.save(model.state_dict(), ckpt_path)

    # end of epoch
print("Training complete.")
