def format_cve(encoded_cve: str) -> str:
    return encoded_cve.split(":")[-1]

def format_edb_id(encoded_cve: str) -> str:
    return f"EDB-{encoded_cve.split(':')[-1]}"

def format_packetstorm_id(encoded_id: str) -> str:
    return f"PACKETSTORM-{encoded_id.split(':')[-1]}"