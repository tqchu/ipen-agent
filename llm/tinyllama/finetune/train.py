import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from llm.tinyllama import loader
from llm.tinyllama.finetune.dataset import QADataset
from llm.tinyllama.finetune.loader import collate_fn

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer, model = loader.initialize_model()
# Assume `model` is the LoRA‐wrapped TinyLlama from Section 3, and it's already .to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)

num_epochs = 3
gradient_accumulation_steps = 1  # or >1 if you want effective larger batch size
save_every = 1000  # steps


train_dataset = QADataset("train_data.jsonl", tokenizer, max_length=512)
val_dataset   = QADataset("val_data.jsonl", tokenizer, max_length=512)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, )
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, )

vocab_size = model.cfg.vocab_size
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

global_step = 0
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    step_loss_accum = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)  # shape (B, T)
        # attention_mask = batch["attention_mask"].to(device)  # not used by TinyLlama
        labels = batch["labels"].to(device)  # shape (B, T)

        # 1) Forward pass: get logits from TinyLlama
        #    TinyLlama.forward returns (logits, new_caches), but we only need logits here
        logits, _ = model(input_ids, kv_caches=None)  # logits: (B, T, V)

        # 2) Shift logits and labels for causal LM loss
        #    - shift_logits[t] predicts token at position t+1
        #    - shift_labels[t]   is the true token at position t+1
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()  # (B, T-1)

        # 3) Compute cross-entropy loss, ignoring positions where shift_labels == -100
        loss = loss_fn(
            shift_logits.view(-1, vocab_size),  # (B*(T-1), V)
            shift_labels.view(-1)  # (B*(T-1),)
        )
        loss = loss / gradient_accumulation_steps
        loss.backward()

        step_loss_accum += loss.item()
        global_step += 1

        # 4) Gradient accumulation / optimizer step
        if global_step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += step_loss_accum
            step_loss_accum = 0.0

        # 5) Update progress bar
        avg_loss = epoch_loss / (global_step // gradient_accumulation_steps or 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

        # 6) Periodic validation & checkpointing
        if global_step % save_every == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vid = vbatch["input_ids"].to(device)  # (B, T)
                    lbl = vbatch["labels"].to(device)  # (B, T)

                    v_logits, _ = model(vid, kv_caches=None)  # (B, T, V)
                    v_shift_logits = v_logits[:, :-1, :].contiguous()  # (B, T-1, V)
                    v_shift_labels = lbl[:, 1:].contiguous()  # (B, T-1)

                    v_loss = loss_fn(
                        v_shift_logits.view(-1, vocab_size),
                        v_shift_labels.view(-1)
                    )
                    val_loss += v_loss.item()
                    val_steps += 1

            avg_val_loss = val_loss / (val_steps or 1)
            print(f"\n→ Step {global_step}: validation loss = {avg_val_loss:.4f}\n")
            model.train()

            # Save adapter (LoRA) weights only
            ckpt_path = f"tinyllama_lora_step{global_step}.pt"
            torch.save(model.state_dict(), ckpt_path)

    # end of epoch
print("Training complete.")