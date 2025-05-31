import torch
import torch.nn as nn
from .rmsnorm import RMSNorm
from .attention import MHA
from .ffn import SwiGLU
from .config import Config
# from rmsnorm import RMSNorm
# from attention import MHA
# from ffn import SwiGLU
# from config import Config

class Block(nn.Module):
    """
    A transformer block containing:
      1. RMSNorm → MHA (with KV‐cache support) → residual
      2. RMSNorm → SwiGLU → residual

    The attention sublayer now accepts and returns a kv_cache dict.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = MHA(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,              # (B, T, D)
        kv_cache: [dict , None] = None  # attention cache { "k":…, "v":… } or None
    ) -> tuple[torch.Tensor, dict]:
        """
        Applies:
          y1 = x + Attn( Norm(x), kv_cache )
          y2 = y1 + SwiGLU( Norm(y1) )
        Returns:
          - out:      the output tensor of shape (B, T, D)
          - new_cache: the updated KV‐cache dict (if kv_cache was provided, or newly built)
        """
        # 1) Attention sublayer
        residual = x
        x_normed = self.attn_norm(x)                   # (B, T, D)
        attn_out, new_cache = self.attn(x_normed, kv_cache)
        # attn_out is (B, T, D) if `kv_cache is None`, otherwise just (B, 1, D) for incremental
        x = residual + attn_out                         # (B, T, D) in full; (B,1,D) per token in incremental

        # 2) Feed‐forward sublayer
        residual2 = x
        x2_normed = self.ffn_norm(x)
        ffn_out = self.ffn(x2_normed)                   # always returns (B, T, D) or (B,1,D)
        out = residual2 + ffn_out

        return out, new_cache

def test_block_kv_cache_full_sequence():
    """
    Verifies that Block (RMSNorm → MHA → residual, then RMSNorm → SwiGLU → residual)
    produces identical outputs when run:
      1. In “batch” mode on prefixes [0..t] (no kv_cache), versus
      2. In “incremental” mode one token at a time carrying the kv_cache forward.

    Because Block(x_prefix, kv_cache=None) returns a tensor of shape (B, prefix_len, D),
    we must explicitly slice off the last token before comparing to the incremental output (B,1,D).
    """

    torch.manual_seed(0)

    # 1) Build config and block
    cfg = Config.from_json("../pretrained_1.1b/config.json")
    block = Block(cfg)
    block.eval()

    B = 2
    D = cfg.d_model
    max_len = 5

    for L in range(1, max_len + 1):
        x = torch.randn(B, L, D)

        # --- Reference: run block on each prefix of length t+1, no cache ---
        reference_outputs = []
        for t in range(L):
            xt_prefix = x[:, : t + 1]  # shape (B, t+1, D)
            with torch.no_grad():
                out_ref_full, _ = block(xt_prefix, kv_cache=None)
                # out_ref_full has shape (B, t+1, D)
            out_ref_last = out_ref_full[:, -1:]  # slice last token → shape (B, 1, D)
            reference_outputs.append(out_ref_last)

        # Concatenate to get (B, L, D)
        full_ref_cat = torch.cat(reference_outputs, dim=1)  # (B, L, D)

        # --- Incremental: feed one token at a time, carrying kv_cache ---
        cache_inc = None
        incremental_outputs = []

        for t in range(L):
            xt = x[:, t : t + 1]  # (B, 1, D)
            with torch.no_grad():
                out_inc_t, cache_inc = block(xt, kv_cache=cache_inc)
                # out_inc_t has shape (B, 1, D)
            incremental_outputs.append(out_inc_t)

            # Stepwise check: token‐t output should match reference_outputs[t]
            torch.testing.assert_close(
                reference_outputs[t],
                out_inc_t,
                atol=1e-4,
                rtol=1e-4,
            )

        # Concatenate incremental outputs → (B, L, D)
        incremental_cat = torch.cat(incremental_outputs, dim=1)

        # Finally assert the whole sequences match
        torch.testing.assert_close(
            full_ref_cat,
            incremental_cat,
            atol=1e-5,
            rtol=1e-5,
        )

        print(f"  ✓ L={L} passed (Block batch vs. incremental match)")

    print("✓ test_block_kv_cache_full_sequence passed for lengths 1..", max_len)


if __name__ == "__main__":
    # You can call both your MHA test (if you kept it here) and this:
    # test_mha_kv_cache_full_sequence()
    test_block_kv_cache_full_sequence()
