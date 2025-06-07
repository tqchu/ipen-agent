import os
import json
import random

from llm.tinyllama.loader import initialize_model

tokenizer, _ = initialize_model()

# Path to your input JSON file and output directory
input_path = "/Users/chutruong/Academic/GraduationThesis/Projects/ipen-agent/data/crawler/msfrpc/data_transform/qa_transformed.json"
current_dir = os.getcwd()
train_output_path = os.path.join(current_dir, "train_minimal_data.jsonl")
val_output_path = os.path.join(current_dir, "val_minimal_data.jsonl")

# Function from previous step to convert a raw record to (prompt, target)
def convert_record_to_string(record):
    question = record["question"].strip()
    answer = record["answer"]

    prompt = (
        "You are a pentest agent. "
        "Given the following state and numbered actions, choose the correct next action.\n\n"
        "You should response with the json format, contains the 'reason' and 'best_action' fields. The best_action contains 'index' and 'module_name' fields.\n\n"
        f"{question}"
    )
    return prompt, answer

# Load all records
with open(input_path, "r") as f:
    records = json.load(f)

# Shuffle and split into 80% train, 20% validation
random.seed(42)  # for reproducibility
random.shuffle(records)
split_idx = int(0.8 * len(records))
train_records = records[:split_idx]
val_records = records[split_idx:]

token_lens = []

# Write the training set
with open(train_output_path, "w") as train_f:
    for rec in train_records:
        prompt, target = convert_record_to_string(rec)
        out_entry = {"instruction": prompt, "response": target}
        train_f.write(json.dumps(out_entry) + "\n")

        # Compute token length for the concatenated prompt+response
        token_count = len(tokenizer.encode(prompt + json.dumps(target, ensure_ascii=False)))
        token_lens.append(token_count)

# Write the validation set
with open(val_output_path, "w") as val_f:
    for rec in val_records:
        prompt, target = convert_record_to_string(rec)
        out_entry = {"instruction": prompt, "response": target}
        val_f.write(json.dumps(out_entry) + "\n")

        # Compute token length for the concatenated prompt+response
        token_count = len(tokenizer.encode(prompt + json.dumps(target, ensure_ascii=False)))
        token_lens.append(token_count)

# After writing all records, print the maximum token length seen
# plot the distribution of token lengths
max_token_length = max(token_lens)
import matplotlib.pyplot as plt

# Assuming you already have a list of token lengths in `token_lens`
plt.figure(figsize=(8, 5))
plt.hist(token_lens, bins=50)
plt.title("Distribution of Token Lengths")
plt.xlabel("Token Count")
plt.ylabel("Number of Examples")
plt.tight_layout()
plt.show()


print(f"Total records: {len(records)}")
print(f"Training set: {len(train_records)} saved to {train_output_path}")
print(f"Validation set: {len(val_records)} saved to {val_output_path}")
