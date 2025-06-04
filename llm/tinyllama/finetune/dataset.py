import json
import torch
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512, pad_id=0):
        """
        Args:
            jsonl_path (str): path to the JSONL file with {"instruction":..., "response":...} lines
            tokenizer (Tok): your tokenizer class, with:
                - tokenizer.encode(text, add_bos:bool, add_eos:bool) -> List[int]
                - tokenizer.bos_id, tokenizer.eos_id
            max_length (int): max sequence length after truncation/padding
            pad_id (int): token ID used for padding (commonly 0)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = pad_id
        self.examples = []

        def pad_or_truncate(id_list):
            """Truncate if longer than max_length, else pad on the right to max_length."""
            if len(id_list) > self.max_length:
                return id_list[: self.max_length]
            else:
                return id_list + [self.pad_id] * (self.max_length - len(id_list))

        with open(jsonl_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                instr = obj["instruction"].strip()   # e.g. "You are a pentest agent. ... Which action next?"
                resp = obj["response"].strip()       # e.g. "1:exploit/aix/local/ibstat_path"

                # 1. Encode the instruction with BOS and EOS
                #    add_bos=True adds the BOS token at the front
                #    add_eos=True  adds the EOS token at the end
                instr_ids = tokenizer.encode(instr, add_bos=True, add_eos=True)
                # 2. Encode the response with only EOS (so we don't re‐insert @bos)
                resp_ids = tokenizer.encode(resp, add_bos=False, add_eos=True)

                # 3. Build the full input sequence: [BOS instr... EOS resp... EOS]
                full_ids = instr_ids + resp_ids

                # 4. Construct labels:
                #    - For each instruction token, set label = -100 (ignore in loss)
                #    - For response tokens (including its EOS), label = token ID
                labels = [-100] * len(instr_ids) + resp_ids.copy()

                # 5. Truncate or pad input_ids and labels to max_length
                input_ids_padded = pad_or_truncate(full_ids)
                labels_padded = pad_or_truncate(labels)

                # 6. Build attention_mask: 1 for non-pad, 0 for pad
                attention_mask = [1 if tid != self.pad_id else 0 for tid in input_ids_padded]

                # 7. Store tensors
                self.examples.append({
                    "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels_padded, dtype=torch.long),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": ex["input_ids"].clone(),
            "attention_mask": ex["attention_mask"].clone(),
            "labels": ex["labels"].clone(),
        }
