# processor.py

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def normalize_platform(platforms):
    """
    The raw 'platform' field might look like:
      ["Msf::Module::Platform::Unix", "Msf::Module::Platform::AIX"]
    We strip off the Ruby namespace and just return ["Unix", "AIX"].
    If that list is empty, return [].
    """
    out = []
    for p in platforms:
        # Split on '::' and take last segment
        parts = p.split("::")
        if len(parts) > 0:
            out.append(parts[-1])
    return out

def normalize_references(raw_refs: List[List[str]]) -> List[Dict[str, str]]:
    """
    raw_refs is a list of two‐element lists: e.g. [ ["CVE","2021-1234"], ["OSVDB","8765"] ]
    Return: [ {"type": "CVE", "id": "2021-1234"}, {"type":"OSVDB", "id":"8765"} ]
    """
    normalized = []
    for pair in raw_refs:
        if len(pair) != 2:
            continue
        ref_type, ref_id = pair
        normalized.append({"type": ref_type, "id": ref_id})
    return normalized

def get_targets(platforms, raw_targets: List[List[Any]]) -> List[Dict[str, Any]]:
    """
    raw_targets comes from MSFRPC as a list of [name, { ... }] entries.
    Each entry has an implicit index (0, 1, ...).
    We’ll flatten into dicts:
      { "index": idx, "name": name, **kwargs }
    """

    normalized_platforms = normalize_platform(platforms)

    out = []

    for target in raw_targets:
        if len(target) < 2:
            continue
        name = target[1]

        if any(platform in str(name).lower() for platform in map(str.lower, normalized_platforms)):
            out.append(name)
    # for target in raw_targets:
    #     if target contains any nor
    #
    return out

def process_single_module(module_path: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given raw JSON from MSFRPC for a single module, return a normalized dict.
    """
    processed = {
        "module_path": module_path.lstrip("/"),  # e.g. "windows/smb/ms08_067_netapi"
        "name": raw.get("Name", ""),
        "description": raw.get("Description", "").strip(),
        "disclosure_date": raw.get("DisclosureDate", ""),
        "authors": raw.get("Author") if isinstance(raw.get("Author"), list) else [raw.get("Author")],
        "license": raw.get("License", ""),
        "references": normalize_references(raw.get("References", [])),
        "platform": raw.get("Platform", ""),
        "arch": raw.get("Arch", []),
        "rank": raw.get("Rank", ""),
        "targets": get_targets(raw.get("Platform", ""), raw.get("Targets", [])),
        "default_options": raw.get("DefaultOptions", {}),
        # Payload section can be nested; capture subfields if present
        "payload": {
            "platforms": raw.get("Payload", {}).get("Platforms", []),
            "archs": raw.get("Payload", {}).get("Archs", []),
            "session_types": raw.get("SessionTypes", []),
            "offsets": raw.get("Payload", {}).get("Offsets", {}),
        },
        # Raw Ruby code for check/exploit; if not present, leave empty
        "check_code": raw.get("Check", ""),
        "exploit_code": raw.get("Exploit", ""),
        # Post‐mixins is typically a list of strings
        "post_mixins": raw.get("Post", []),
    }

    # Sanity: Authors might be a single string; coerce to list
    if processed["authors"] is None:
        processed["authors"] = []
    elif isinstance(processed["authors"], str):
        processed["authors"] = [processed["authors"]]

    return processed

def process_bulk(raw_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    raw_data is a dict mapping {module_path: raw_json}
    Return a list of processed module dicts.
    """
    processed_list = []
    for module_path, raw in raw_data.items():
        try:
            proc = process_single_module(module_path, raw)
            processed_list.append(proc)
        except Exception as e:
            logger.error(f"Failed to process module {module_path}: {e}")
    logger.info(f"Processed {len(processed_list)} modules out of {len(raw_data)} crawled")
    return processed_list
