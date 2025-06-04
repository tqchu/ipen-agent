import torch
from torch.utils.data import DataLoader

from llm.tinyllama import loader
from llm.tinyllama.finetune.dataset import QADataset


def collate_fn(batch):
    input_ids = [ex["input_ids"] for ex in batch]
    attention_mask = [ex["attention_mask"] for ex in batch]
    labels = [ex["labels"] for ex in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenizer, _ = loader.initialize_model()


