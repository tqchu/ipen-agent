"""Broadcast‑safe Rotary Positional Embedding (RoPE)."""
import torch

__all__ = ["rope_cache", "apply_rope"]

def rope_cache(seq_len: int, dim: int, theta: float, device: torch.device):
    """
    Build rotary‐embedding sine / cosine caches that match HF Llama:
      sin, cos: shape (1, 1, seq_len, dim), where dim == head_dim
    """
    half = dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)             # (seq_len, half)
    emb = torch.cat([freqs, freqs], dim=-1)      # (seq_len, dim)
    sin = emb.sin()[None, None, :, :]            # (1,1,seq_len,dim)
    cos = emb.cos()[None, None, :, :]
    return sin, cos

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """
    x   : (B, H, T, D)
    sin : (1, 1, T, D)
    cos : (1, 1, T, D)

    Returns x * cos + rotate_half(x) * sin
    """
    # ensure sin/cos cover full head_dim
    # (rope_cache already returns full-width tables)
    return x * cos + rotate_half(x) * sin
