#!/usr/bin/env python3
import json
import os
import json
import sys


def format_question(state: dict, available_actions: list) -> str:
    """
    Build a concise prompt string given the 'state' dict and 'available_actions' list.
    """
    vuln = state.get("vulnerabilities", "unknown_vulnerability")
    platform = state.get("platform", "unknown_platform")
    targets = state.get("targets", "unknown_targets")

    lines = [f'Current state: a {platform} host ({", ".join(targets)}) has been identified with vulnerabilities {", ".join(vuln)}.',
             "Available actions:"]
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


def format_answer(next_action: dict, reason: str) -> dict:
    """
    Build the 'answer' JSON object with keys:
      - "reason": (empty string placeholder)
      - "best_action": { "index": X, "module_name": "..." }
    """
    return {
        "reason": reason,
        "best_action": {
            "index": next_action.get("index"),
            "module_name": next_action.get("module_path")
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
            answer = format_answer(next_action, record.get("reason", ""))

            qa_list.append({
                "question": question,
                "answer": answer
            })

    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(qa_list, fout, indent=2)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_jsonl = os.path.join(current_dir, "preprocessed.json")
    output_jsonl = os.path.join(current_dir, "qa_transformed.json")
    transform_jsonl_to_list(input_jsonl, output_jsonl)

    print(f"Transformed '{input_jsonl}' into question/answer pairs in '{output_jsonl}'.")
