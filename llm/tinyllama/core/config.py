"""TinyLlama configuration dataclass.

The public `config.json` on HuggingFace follows the Llama‑2 naming scheme.
This helper maps those keys → the concise internal names we use in code.
"""
from dataclasses import dataclass
import json
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Config:
    vocab_size: int
    max_seq_len: int  # was "max_position_embeddings" in HF JSON
    d_model: int  # was "hidden_size"
    n_heads: int  # was "num_attention_heads"
    n_kv_heads: int  # was "num_key_value_heads"
    n_layers: int  # was "num_hidden_layers"
    d_ff: int  # was "intermediate_size"
    rope_theta: float  # was "rope_theta"
    rms_eps: float  # was "rms_norm_eps"
    attention_bias: Optional[bool]
    bos_token_id: Optional[int]
    eos_token_id: Optional[int]
    hidden_act: Optional[str]
    initializer_range: Optional[float]
    model_type: Optional[str]
    pretraining_tp: Optional[int]
    rope_scaling: Optional[Any]
    tie_word_embeddings: Optional[bool]
    torch_dtype: Optional[str]
    transformers_version: Optional[str]
    use_cache: Optional[bool]

    # (store the raw JSON here so we can do get(key) later)
    j: Dict[str, Any]

    @classmethod
    def from_json(cls, path: str):
        """Load an HF‐style JSON and populate every field (renaming where needed)."""
        with open(path, "r") as f:
            j = json.load(f)

        return cls(
            # ――――― fields with renaming/mapping ―――――
            vocab_size=j["vocab_size"],
            max_seq_len=j["max_position_embeddings"],
            d_model=j["hidden_size"],
            n_heads=j["num_attention_heads"],
            n_kv_heads=j.get("num_key_value_heads", j["num_attention_heads"] // 2),
            n_layers=j["num_hidden_layers"],
            d_ff=j["intermediate_size"],
            rope_theta=j.get("rope_theta", 10000.0),
            rms_eps=j.get("rms_norm_eps", 1e-5),

            # ――――― new fields (pull straight from JSON or default to None) ―――――
            attention_bias=j.get("attention_bias"),
            bos_token_id=j.get("bos_token_id"),
            eos_token_id=j.get("eos_token_id"),
            hidden_act=j.get("hidden_act"),
            initializer_range=j.get("initializer_range"),
            model_type=j.get("model_type"),
            pretraining_tp=j.get("pretraining_tp"),
            rope_scaling=j.get("rope_scaling"),
            tie_word_embeddings=j.get("tie_word_embeddings"),
            torch_dtype=j.get("torch_dtype"),
            transformers_version=j.get("transformers_version"),
            use_cache=j.get("use_cache"),

            # ――――― store the raw JSON dict for any adhoc lookups ―――――
            j=j,
        )

    def get(self, key: str) -> Any:
        """Get a value from the original JSON dict (or return None if missing)."""
        return self.j.get(key)
