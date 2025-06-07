"""Rename HuggingFace TinyLlama keys -> our PyTorch skeleton names."""
import torch, os, re
from collections import OrderedDict

_REPLACERS = [
    (r"^model\.embed_tokens", "emb"),
    (r"\.self_attn\.", ".attn."),
    (r"\.q_proj", ".q_proj"),
    (r"\.k_proj", ".k_proj"),
    (r"\.v_proj", ".v_proj"),
    (r"\.o_proj", ".o_proj"),
    (r"\.input_layernorm", ".attn_norm"),
    (r"\.post_attention_layernorm", ".ffn_norm"),
    (r"\.mlp\.gate_proj", ".ffn.gate"),
    (r"\.mlp\.up_proj", ".ffn.up"),
    (r"\.mlp\.down_proj", ".ffn.down"),
    (r"\.norm$", ".norm"),
    (r"^model\.norm", "norm"),
    (r"^lm_head\.bias$", "final_logits_bias"),  # HF root bias
    (r"^model\.lm_head\.bias$", "final_logits_bias"),
]

def _rename(key: str) -> str:
    key = key.replace("model.layers.", "blocks.")

    for pat, repl in _REPLACERS:
        key = re.sub(pat, repl, key)
    return key


def load_state(path: str):
    """Load all *.bin or *.safetensors shards, rename keys, return one state_dict."""
    import safetensors.torch as st
    state = OrderedDict()

    for fname in sorted(os.listdir(path)):
        if fname.endswith((".bin", ".safetensors")):
            fpath = os.path.join(path, fname)
            shard = st.load_file(fpath) if fname.endswith(".safetensors") else \
                torch.load(fpath, map_location="cpu")
            for k, v in shard.items():
                state[_rename(k)] = v
    return state


def load_state_pt(pt_path: str) -> dict:
    """
    Load a PyTorch state_dict from a single .pt (or .bin) file, then rename keys.

    Args:
        pt_path: Path to the .pt file containing a raw state_dict.

    Returns:
        An OrderedDict with all keys renamed via `_rename(...)`.
    """
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {pt_path}")

    raw_state = torch.load(pt_path, map_location="cpu")
    renamed_state = OrderedDict()
    for k, v in raw_state.items():
        renamed_state[_rename(k)] = v
    return renamed_state
