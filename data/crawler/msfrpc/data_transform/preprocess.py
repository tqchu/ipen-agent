import json
import os
import random
import sys


def extract_cves(references):
    """
    Given a list of reference dicts (each with fields 'type' and 'id'),
    return a list of CVE IDs in that list (e.g. ["CVE-2013-4011", ...]).
    """
    return [f'{ref["type"]}-{ref.get("id")}' for ref in references if ref.get("type") != "URL"]


# def get_target(targets):
#     """
#     The raw 'platform' field might look like:
#       ["Msf::Module::Platform::Unix", "Msf::Module::Platform::AIX"]
#     We strip off the Ruby namespace and just return ["Unix", "AIX"].
#     If that list is empty, return [].
#     """
#     out = []
#     for p in targets:
#         # Split on '::' and take last segment
#         parts = p.split("::")
#         if len(parts) > 0:
#             out.append(parts[-1])
#     return out

def normalize_platform(targets):
    """
    The raw 'platform' field might look like:
      ["Msf::Module::Platform::Unix", "Msf::Module::Platform::AIX"]
    We strip off the Ruby namespace and just return ["Unix", "AIX"].
    If that list is empty, return [].
    """
    out = []
    for p in targets:
        # Split on '::' and take last segment
        parts = p.split("::")
        if len(parts) > 0:
            out.append(parts[-1])
    return out

def infer_exploit_type(module_path):
    """
    Infer a high-level exploit type based on the second component of module_path.
    Examples:
      - "linux/http/xyz"   → "remote_http_exploit"
      - "windows/local/abc" → "local_priv_esc"
      - "freebsd/misc/def"  → "remote_unix_exploit"
    """
    parts = module_path.split('/')
    if len(parts) >= 2:
        category = parts[1].lower()
        if category == "http":
            return "remote_http_exploit"
        if category == "local":
            return "local_priv_esc"
        if category in ("linux", "freebsd", "unix", "aix"):
            return "remote_unix_exploit"
        if category == "windows":
            return "windows_exploit"
    return "generic_exploit"

def canonical_action(obj):
    """
    Given a Metasploit module_path (e.g. "windows/smb/ms17_010_eternalblue"),
    return the "canonical" Metasploit command‐string (with a placeholder for RHOST).
    """
    # We assume a convention:
    #   use exploit/<module_path>; set RHOSTS <target_ip>; exploit
    module_path = obj.get("module_path")
    platforms_supported = normalize_platform(obj.get("platform"))

    if obj.get("targets"):
        platforms_supported = f'{", ".join(platforms_supported)}, {", ".join(obj.get("targets"))}'

    cves = extract_cves(obj.get("references", []))
    exploit_type = infer_exploit_type(module_path)

    prereqs = []
    for stype in obj.get("payload", {}).get("session_types", []):
        prereqs.append(f"requires_session_type:{stype}")

    # Command template pattern
    cmd_template = f"use {module_path}; set RHOSTS {{target_ip}}; exploit"

    return {
        "module_path": module_path,
        "platforms_supported": platforms_supported,
        "type": exploit_type,
        "prerequisites": prereqs,
        "associated_vuln": cves,
        "command_template": cmd_template
    }

def correct_action_string(idx, module_path):
    """
    Given an index (1-based) and a module_path, return the formatted action string.
    Example: idx=1, module_path="windows/smb/ms17_010_eternalblue"
    returns "1:use exploit/windows/smb/ms17_010_eternalblue; set RHOSTS {{target_ip}}; exploit"
    """
    return {
        "index": idx,
        "module_path": module_path,
    }

def build_training_example(raw_obj, all_actions, distractor_count=2):
    """
    raw_obj: a single JSON‐loaded dict describing one Metasploit module.
    all_actions: list of all canonical_action(...) strings for every module in the dump.
    distractor_count: how many "wrong" actions to sample alongside the correct one.

    Returns a dict of the form:
      {
        "state": { ... },
        "available_actions": [ ... ],
        "next_action": "...",
        "source": "<module_path>"
      }
    """
    module_path = raw_obj.get("module_path")
    if not module_path:
        return None

    # 1) Extract CVEs
    cve_list = extract_cves(raw_obj.get("references", []))
    chosen_vulners = cve_list

    # 2) Extract a primary platform (e.g. "Windows", "Unix", "AIX", "Linux", etc.)
    platform_list = normalize_platform(raw_obj.get("platform", []))
    chosen_platform = platform_list[0] if len(platform_list) > 0 else None

    # 3) Build the "state" dict
    state = {}
    if chosen_vulners:
        state["vulnerabilities"] = chosen_vulners
    if chosen_platform:
        state["platform"] = chosen_platform

    state["targets"] = raw_obj.get("targets")

    # 4) Build the "correct" action string
    correct_action = canonical_action(raw_obj)

    # 5) Sample distractors (ensure we don't pick the correct action again)
    #    We want N random other actions from all_actions, excluding correct_action.
    distractors = []
    if len(all_actions) > 1:
        pool = [a for a in all_actions if a != correct_action]
        # If the dump is huge, sampling without replacement is okay.
        # If pool is smaller than distractor_count, just use entire pool.
        sample_size = min(distractor_count, len(pool))
        distractors = random.sample(pool, sample_size)

    # 6) Build available_actions = [ correct_action, *distractors ], then shuffle
    available = [correct_action] + distractors
    random.shuffle(available)

    idx = available.index(correct_action)

    correct_action_st = correct_action_string(idx, module_path)

    # 7) Build the reason string
    # get the reason from the description, if description is empty, use the name
    reason = raw_obj.get("description", "").strip()
    if not reason:
        reason = raw_obj.get("name", "").strip()

    if reason:  # Check if reason is not an empty string
        reason = reason[0].lower() + reason[1:]

    # 7) Build final example
    example = {
        "state": state,
        "available_actions": available,
        "next_action": correct_action_st,
        "reason": "Because " + reason,
        "source": module_path
    }
    return example


def main(input_jsonl_path, output_jsonl_path, distractor_count=2, seed=42):
    """
    1) Read all Metasploit module JSONs from input_jsonl_path (one JSON per line).
    2) Build a list of all possible 'correct' actions.
    3) Stream again, convert each line into a training example, and write to output_jsonl_path.

    usage:
      python convert_metasm_to_finetune.py metasploit_raw.jsonl metasploit_ft.jsonl
    """
    random.seed(seed)

    # 1) First pass: collect every module_path → canonical_action
    all_actions = []
    with open(input_jsonl_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            module_path = obj.get("module_path")
            if not module_path:
                continue
            all_actions.append(canonical_action(obj))

    # 2) Second pass: build training examples
    with open(input_jsonl_path, "r", encoding="utf-8") as f_in, \
            open(output_jsonl_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                raw_obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            example = build_training_example(raw_obj, all_actions, distractor_count)
            if example is None:
                continue

            # Write it out as one JSON per line
            f_out.write(json.dumps(example))
            f_out.write("\n")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "../data/all_modules.jsonl")
    output_path = os.path.join(current_dir, "preprocessed.json")

    main(input_path, output_path, distractor_count=1)
