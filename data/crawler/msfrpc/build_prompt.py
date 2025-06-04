#!/usr/bin/env python3
import json
import os
import json
import sys


def format_question(state: dict, available_actions: list) -> str:
    """
    Build a concise prompt string given the 'state' dict and 'available_actions' list.
    """
    vuln = state.get("vulnerability", "unknown_vulnerability")
    platform = state.get("platform", "unknown_platform")

    lines = [f"Current state: a {platform} host has been identified with vulnerability {vuln}."]
    lines.append("Available actions:")
    for idx, action in enumerate(available_actions, start=1):
        lines.append(f"{idx}. {action}")
    lines.append("\nWhich action should be executed next?")

    return "\n".join(lines)


def extract_module_name(msf_action: str) -> str:
    """
    Given a Metasploit 'use ...; set RHOSTS ...; exploit' string,
    extract just the module path (e.g. 'exploit/aix/local/ibstat_path').
    """
    msf_action = msf_action.strip()
    if not msf_action.startswith("use "):
        return msf_action
    remainder = msf_action[len("use "):]
    module_and_rest = remainder.split(";", 1)[0].strip()
    return module_and_rest


def format_answer(available_actions: list, next_action: str) -> dict:
    """
    Build the 'answer' JSON object with keys:
      - "reason": (empty string placeholder)
      - "best_action": { "index": X, "module_name": "..." }
    """
    try:
        idx0 = available_actions.index(next_action)
        one_based = idx0 + 1
    except ValueError:
        one_based = -1

    module_name = extract_module_name(next_action)
    return {
        "reason": "",
        "best_action": {
            "index": one_based,
            "module_name": module_name
        }
    }


def transform_jsonl_to_list(in_path: str, out_path: str):
    """
    Read the input JSONL, transform each line into a {question, answer} pair,
    collect them in a list, and write the list as a single JSON array to out_path.
    """
    qa_list = []
    with open(in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            state = record.get("state", {})
            available_actions = record.get("available_actions", [])
            next_action = record.get("next_action", "")

            question = format_question(state, available_actions)
            answer = format_answer(available_actions, next_action)

            qa_list.append({
                "question": question,
                "answer": answer
            })

    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(qa_list, fout, indent=2)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_jsonl = os.path.join(current_dir, "json_prompt.jsonl")
    output_jsonl = os.path.join(current_dir, "qa_transformed.json")
    transform_jsonl_to_list(input_jsonl, output_jsonl)

    print(f"Transformed '{input_jsonl}' into question/answer pairs in '{output_jsonl}'.")
