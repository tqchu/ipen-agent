import os
import json
import random

# Path to your input JSON file and output directory
input_path = "/Users/chutruong/Academic/GraduationThesis/Projects/ipen-agent/data/crawler/msfrpc/qa_transformed.json"
current_dir = os.getcwd()
train_output_path = os.path.join(current_dir, "train_data.jsonl")
val_output_path = os.path.join(current_dir, "val_data.jsonl")

# Function from previous step to convert a raw record to (prompt, target)
def convert_record_to_string(record):
    question = record["question"].strip()
    best = record["answer"]["best_action"]
    idx = best["index"]
    module = best["module_name"]
    target = f"{idx}:{module}"
    prompt = (
        "You are a pentest agent. "
        "Given the following state and numbered actions, choose the correct next action.\n\n"
        f"{question}"
    )
    return prompt, target

# Load all records
with open(input_path, "r") as f:
    records = json.load(f)

# Shuffle and split into 80% train, 20% validation
random.seed(42)  # for reproducibility
random.shuffle(records)
split_idx = int(0.8 * len(records))
train_records = records[:split_idx]
val_records = records[split_idx:]

# Write the training set
with open(train_output_path, "w") as train_f:
    for rec in train_records:
        prompt, target = convert_record_to_string(rec)
        out_entry = {"instruction": prompt, "response": target}
        train_f.write(json.dumps(out_entry) + "\n")

# Write the validation set
with open(val_output_path, "w") as val_f:
    for rec in val_records:
        prompt, target = convert_record_to_string(rec)
        out_entry = {"instruction": prompt, "response": target}
        val_f.write(json.dumps(out_entry) + "\n")

print(f"Total records: {len(records)}")
print(f"Training set: {len(train_records)} saved to {train_output_path}")
print(f"Validation set: {len(val_records)} saved to {val_output_path}")
