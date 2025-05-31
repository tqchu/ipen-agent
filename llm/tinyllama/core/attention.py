import torch
from torch import nn

from llm.tinyllama.core.config import Config
from llm.tinyllama.core.rotary import apply_rope, rope_cache
import torch.nn.functional as F

class MHA(nn.Module):
    """
    Multi-Head Attention with
    • Rotary positional embedding
    • Torch-native Scaled-Dot-Product-Attention
    • Optional key/value cache for fast autoregressive decoding
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H, HK = cfg.n_heads, cfg.n_kv_heads
        self.d_head = cfg.d_model // H

        self.q_proj = nn.Linear(cfg.d_model, H  * self.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, HK * self.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, HK * self.d_head, bias=False)
        self.o_proj = nn.Linear(H * self.d_head, cfg.d_model, bias=False)

        self.sin = self.cos = None            # RoPE cache

    # ---------- rotary helpers (unchanged) ----------
    def rope(self, q, k, *, pos=None):
        B, H, T, D = q.shape
        T_cache = pos if pos is not None else T
        sin, cos = self.build_rope(T_cache, q.device)
        sin, cos = sin[..., :T, :], cos[..., :T, :]
        return apply_rope(q, sin, cos), apply_rope(k, sin, cos)

    def build_rope(self, T, device):
        if self.sin is None or self.sin.size(1) < T:
            self.sin, self.cos = rope_cache(T, self.d_head,
                                            self.cfg.rope_theta, device)
        return self.sin[:, :T], self.cos[:, :T]

    # ---------- forward with KV cache ----------
    def forward(self, x: torch.Tensor,
                kv_cache: [dict , None] = None) -> tuple[torch.Tensor, dict]:
        """
        x        : (B, T, D)  – usually T==1 during generation
        kv_cache : None (first call) or {'k': (B,H,Tprev,Dh), 'v': …}

        returns   y          – (B, T, D)
                  new_cache  – structure identical to kv_cache
        """
        B, T, _  = x.shape
        H, HK, Dh = self.cfg.n_heads, self.cfg.n_kv_heads, self.d_head

        # Project current tokens
        q = self.q_proj(x).view(B, T, H , Dh).transpose(1, 2)      # (B,H,T,Dh)
        k = self.k_proj(x).view(B, T, HK, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, HK, Dh).transpose(1, 2)

        # Rotary position encoding for new pieces only
        q, k = self.rope(q, k)

        # Concatenate with cache (if any)
        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)               # time axis
            v = torch.cat([kv_cache["v"], v], dim=2)

        # Prepare new cache (detach so grads don’t flow in eval mode)
        new_cache = {"k": k.detach(), "v": v.detach()}

        # Expand KV heads to full head count
        repeat = H // HK
        k_full = k.repeat_interleave(repeat, dim=1)
        v_full = v.repeat_interleave(repeat, dim=1)

        # ✨ single fused kernel — causal mask handled internally
        out = F.scaled_dot_product_attention(q, k_full, v_full,
                                             attn_mask=None,
                                             is_causal=True)      # (B,H,T,Dh)

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out), new_cache

def test_mha_kv_cache():
    torch.manual_seed(0)

    # ----- tiny dummy config -----
    cfg = Config(
        d_model=64, n_heads=4, n_kv_heads=2,
        rope_theta=10000,   # whatever your helper expects
        # … other irrelevant fields
    )
    mha = MHA(cfg)
    mha.eval()

    B, L = 1, 8                       # 8-token prefix
    x    = torch.randn(B, L, cfg.d_model)

    # 1) full pass – no cache
    ref_out, _ = mha(x, kv_cache=None)

    # 2) incremental pass – build cache one token at a time
    cache   = None
    outputs = []
    for t in range(L):
        y, cache = mha(x[:, t:t+1], kv_cache=cache)
        outputs.append(y)
    inc_out = torch.cat(outputs, dim=1)

    # ----- compare -----
    torch.testing.assert_close(ref_out, inc_out, atol=1e-5, rtol=1e-4)
    print("✓ KV-cache produces identical output")

if __name__ == "__main__":
    test_mha_kv_cache()
