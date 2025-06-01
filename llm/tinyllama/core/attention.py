import csv
import math
import os

import torch
from torch import nn

from llm.tinyllama.core.config import Config
from llm.tinyllama.core.rotary import apply_rope, rope_cache
import torch.nn.functional as F

batch_csv = "debug_batch.csv"
inc_csv = "debug_inc.csv"

class MHA(nn.Module):
    """
    Multi-Head Attention with
      • rotary embedding
      • torch SDPA kernel
      • key/value cache for fast decoding

    BEHAVIOR CHANGES:

      * Batch (kv_cache=None) → returns full (B, T, D) output
      * Incremental (kv_cache!=None) → returns only last step (B, 1, D)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H, HK = cfg.n_heads, cfg.n_kv_heads
        self.d_head = cfg.d_model // H

        self.q_proj = nn.Linear(cfg.d_model, H * self.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, HK * self.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, HK * self.d_head, bias=False)
        self.o_proj = nn.Linear(H * self.d_head, cfg.d_model, bias=False)

        # We'll cache sin/cos tables up to the largest needed sequence length:
        self.sin = None  # shape will be (1, 1, max_seq_len, d_head)
        self.cos = None

    def _build_rope(self, T: int, device: torch.device):
        """
        Ensure that self.sin/self.cos are at least length T along the time dimension.
        Returns (sin[:, :, :T, :], cos[:, :, :T, :]) each of shape (1,1,T,d_head).
        """
        # If we haven't built a table yet, or our current table is too short, rebuild:
        if self.sin is None or self.sin.size(2) < T:
            self.sin, self.cos = rope_cache(T, self.d_head, self.cfg.rope_theta, device)

        # Return the first T positions:
        return self.sin[:, :, :T, :], self.cos[:, :, :T, :]

    def forward(
        self,
        x: torch.Tensor,            # (B, T, D)
        kv_cache: [dict, None] = None
    ) -> tuple[torch.Tensor, dict]:
        B, T, _ = x.shape
        H, HK, Dh = self.cfg.n_heads, self.cfg.n_kv_heads, self.d_head

        # ── 1) Linear projections ──────────────────────────────────────────
        q_all = (
            self.q_proj(x.contiguous())
            .view(B, T, H, Dh)
            .transpose(1, 2)
        )  # (B, H, T, Dh)
        k_raw = (
            self.k_proj(x.contiguous())
            .view(B, T, HK, Dh)
            .transpose(1, 2)
        )  # (B, HK, T, Dh)
        v_raw = (
            self.v_proj(x.contiguous())
            .view(B, T, HK, Dh)
            .transpose(1, 2)
        )  # (B, HK, T, Dh)

        # ── 2) Build RoPE tables & slice out the T positions ──────────────
        offset = 0 if kv_cache is None else kv_cache["k"].size(2)
        sin_all, cos_all = self._build_rope(offset + T, x.device)
        sin_slice = sin_all[:, :, offset : offset + T, :]  # (1,1,T,Dh)
        cos_slice = cos_all[:, :, offset : offset + T, :]  # (1,1,T,Dh)

        # ── 3) Apply RoPE to queries & raw keys ───────────────────────────
        q_all = apply_rope(q_all, sin_slice, cos_slice)      # (B, H, T, Dh)
        k_raw = apply_rope(k_raw, sin_slice, cos_slice)      # (B, HK, T, Dh)

        # ── 4) Concatenate new K/V with cache (if any) ────────────────────
        if kv_cache is not None:
            k_raw = torch.cat([kv_cache["k"], k_raw], dim=2)  # (B,HK,prev+T,Dh)
            v_raw = torch.cat([kv_cache["v"], v_raw], dim=2)  # (B,HK,prev+T,Dh)

        # Build new cache
        new_cache = {
            "k": k_raw.detach().contiguous().clone(),
            "v": v_raw.detach().contiguous().clone(),
        }

        # ── 5) Expand HK→H via repeat_interleave ──────────────────────────
        repeat = H // HK
        k_full = k_raw.contiguous().repeat_interleave(repeat, dim=1)  # (B,H,T_full,Dh)
        v_full = v_raw.contiguous().repeat_interleave(repeat, dim=1)  # (B,H,T_full,Dh)

        # ── 6) Split into batch vs. incremental attention paths ──────────
        if kv_cache is None:
            # — Batch mode: attend to all T queries in one shot —
            q = q_all.contiguous()      # (B,H,T,Dh)
            k = k_full.contiguous()     # (B,H,T,Dh)
            v = v_full.contiguous()     # (B,H,T,Dh)

            # ── ▣ DEBUG #3: Compute raw attention scores for last query ▣ ──
            inv_sqrt = 1.0 / math.sqrt(Dh)
            scores_full = torch.einsum("b h q d, b h k d -> b h q k", q, k) * inv_sqrt
            scores_ref_last = scores_full[:, :, -1:, :]  # shape (B,H,1,T)

            probs_full = F.softmax(scores_ref_last, dim=-1)  # still (B,H,1,T)

            out_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

            # ── ▣ DEBUG #4: Print attention output for last query before o_proj ▣ ──
            attn_ref_last = out_attn[:, :, -1:, :]  # (B,H,1,Dh)

            attn_out_vec_batch = attn_ref_last[0, 0, 0, :].tolist()

            attn_last3 = attn_ref_last[0, 0, 0, :3].tolist()

            # Log attn output shape and first 3 dims
            shape_out_attn = tuple(out_attn.shape)

            # Reshape & project → (B,T,H*Dh) → (B,T,D)
            out = (
                out_attn
                .transpose(1, 2)       # (B, T, H, Dh)
                .contiguous()
                .view(B, T, H * Dh)
                .contiguous()
            )
            out = self.o_proj(out)    # (B, T, D)

            return out, new_cache

        else:
            # — Incremental mode: only one new query is present —
            q = q_all.contiguous()     # (B, H, 1, Dh)
            k = k_full.contiguous()    # (B, H, T_full, Dh)
            v = v_full.contiguous()    # (B, H, T_full, Dh)

            B, H, _, Dh = q.shape
            T_full = k.size(2)  # number of keys so far

            prefix = torch.zeros((B, H, T_full - 1, Dh), device=q.device, dtype=q.dtype)
            q_padded = torch.cat([prefix, q], dim=2)

            out_attn = F.scaled_dot_product_attention(
                q_padded,  # (B, H, T_full, Dh)
                k,  # (B, H, T_full, Dh)
                v,  # (B, H, T_full, Dh)
                attn_mask=None,
                is_causal=True
            )

            out_last = out_attn[:, :, -1:, :]  # shape = (B, H, 1, Dh)

            out = out_last.transpose(1, 2).contiguous().view(B, 1, H * Dh)
            out = self.o_proj(out)  # (B, 1, D)
            return out, new_cache

def reset_debug_csvs():
    global batch_csv, inc_csv

    for fname in [batch_csv, inc_csv]:
        # Overwrite existing file or create a new one with only the header
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["var_name", "value"])

def test_mha_kv_cache_full_sequence():
    global batch_csv, inc_csv

    torch.manual_seed(0)

    # Build the model
    cfg = Config.from_json("../pretrained_1.1b/config.json")
    mha = MHA(cfg)
    mha.eval()

    B = 2
    D = cfg.d_model
    max_len = 5

    for L in range(2, max_len + 1):
        batch_csv = "debug_batch.csv"

        reset_debug_csvs()

        with open("debug_batch_with_cache.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["var_name", "value"])

        x = torch.randn(B, L, D)

        # --- (1) REFERENCE: run batch on each prefix [0..t], no cache ---
        reference_outputs = []
        for t in range(L):
            with torch.no_grad():
                out_ref_full, _ = mha(x[:, : t + 1], kv_cache=None)
                # out_ref_full has shape (B, t+1, D)
            # Slice off ONLY the last row → shape (B,1,D)
            out_ref_t = out_ref_full[:, -1 : , :]  # (B,1,D)
            reference_outputs.append(out_ref_t)

        # Combine them → (B, L, D)
        full_ref_cat = torch.cat(reference_outputs, dim=1)

        # --- (2) INCREMENTAL: feed tokens one by one, carrying kv_cache ---
        cache_inc = None
        incremental_outputs = []

        batch_csv = "debug_batch_with_cache.csv"

        for t in range(L):
            with torch.no_grad():
                out_inc_t, cache_inc = mha(x[:, t : t + 1], kv_cache=cache_inc)
                # out_inc_t has shape (B,1,D)
            # Stepwise assertion: they must match exactly
            print("For i = ", t)
            torch.testing.assert_close(
                reference_outputs[t],
                out_inc_t,
                atol=1e-4,
                rtol=1e-4,
            )
            incremental_outputs.append(out_inc_t)

        # Now concatenate → (B, L, D)
        incremental_cat = torch.cat(incremental_outputs, dim=1)

        # Final assertion: the entire sequences must match
        torch.testing.assert_close(
            full_ref_cat,
            incremental_cat,
            atol=1e-5,
            rtol=1e-5,
        )

        print(f"  ✓ L={L} passed (reference vs. incremental match)")

    print("✓ test_mha_kv_cache_full_sequence passed for lengths 1..", max_len)

if __name__ == "__main__":
    test_mha_kv_cache_full_sequence()
