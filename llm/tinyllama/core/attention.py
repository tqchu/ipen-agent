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
        # Ensure CSV exists; if not, write header
        if not os.path.isfile(batch_csv):
            with open(batch_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["var_name", "value"])

        if not os.path.isfile(inc_csv):
            with open(inc_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["var_name", "value"])

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

        # Right after you compute q_all = self.q_proj(x).view(...).transpose(...)
        if kv_cache is None:
            # Batch mode
            if T > 1:
                # Grab the pre‐RoPE projection of token 1 (the second token in the batch)
                token1_pre_rope = q_all[:, :, 1, :]  # shape = (B, H, Dh)
                with open(batch_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["batch_token1_pre_rope", token1_pre_rope.tolist()])
            else:
                # T == 1: there *is* no token 1 yet, so skip this
                with open(batch_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["batch_token1_pre_rope", "N/A (only T=1)"])

        else:
            # Incremental mode
            # Here, q_all has shape (B, H, 1, Dh).  The “new” token is always index 0.
            token1_pre_rope = q_all[:, :, 0, :]  # shape = (B, H, Dh)
            with open(inc_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["inc_token1_pre_rope", token1_pre_rope.tolist()])

        # ── 2) Build RoPE tables & slice out the T positions ──────────────
        offset = 0 if kv_cache is None else kv_cache["k"].size(2)
        sin_all, cos_all = self._build_rope(offset + T, x.device)
        sin_slice = sin_all[:, :, offset : offset + T, :]  # (1,1,T,Dh)
        cos_slice = cos_all[:, :, offset : offset + T, :]  # (1,1,T,Dh)

        # ── 3) Apply RoPE to queries & raw keys ───────────────────────────
        q_all = apply_rope(q_all, sin_slice, cos_slice)      # (B, H, T, Dh)
        k_raw = apply_rope(k_raw, sin_slice, cos_slice)      # (B, HK, T, Dh)
        # v_raw remains unrotated.

        # ── ▣ DEBUG #1: Print RoPE‐rotated Q for token index 1 ▣ ───────────
        # if x.shape[1] == 2 and kv_cache is None:
        #     # Batch mode on a 2-token prefix → print last query slice
        #     print("[DEBUG Q-BATCH] q_all_last (heads x first3 dims):", q_all[:, :, -1:, :3])
        # if x.shape[1] == 1 and kv_cache is not None and kv_cache["k"].shape[2] == 1:
        #     # Incremental, second token → q_all is shape (B,H,1,Dh)
        #     print("[DEBUG Q-INC]   q_all (heads x first3 dims):", q_all[:, :, :, :3])

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

        # ── ▣ DEBUG #2b: Print concatenated V for both tokens ▣ ───────────
        # if x.shape[1] == 2 and kv_cache is None:
        #     # Batch: two‐token prefix → v_full has shape (B, H, 2, Dh)
        #          print("[DEBUG V-BATCH] v_full shape:", v_full.shape)
        #          print("[DEBUG V-BATCH] v_full[:,:,0,:3]:", v_full[:, :, 0, :3])
        #          print("[DEBUG V-BATCH] v_full[:,:,1,:3]:", v_full[:, :, 1, :3])
        # if x.shape[1] == 1 and kv_cache is not None and kv_cache["v"].shape[2] == 1:
        #     # Incremental second step → v_full also (B, H, 2, Dh)
        #          print("[DEBUG V-INC]   v_full shape:", v_full.shape)
        #          print("[DEBUG V-INC]   v_full[:,:,0,:3]:", v_full[:, :, 0, :3])
        #          print("[DEBUG V-INC]   v_full[:,:,1,:3]:", v_full[:, :, 1, :3])

        # ── ▣ DEBUG #2: Print concatenated K for both tokens ▣ ───────────
        # if x.shape[1] == 2 and kv_cache is None:
        #     # Batch: two‐token prefix
        #     print("[DEBUG K-BATCH] k_full shape:", k_full.shape)
        #     print("[DEBUG K-BATCH] k_full[:,:,0,:3]:", k_full[:, :, 0, :3])
        #     print("[DEBUG K-BATCH] k_full[:,:,1,:3]:", k_full[:, :, 1, :3])
        # if x.shape[1] == 1 and kv_cache is not None and kv_cache["k"].shape[2] == 1:
        #     # Incremental second step
        #     print("[DEBUG K-INC]   k_full shape:", k_full.shape)
        #     print("[DEBUG K-INC]   k_full[:,:,0,:3]:", k_full[:, :, 0, :3])
        #     print("[DEBUG K-INC]   k_full[:,:,1,:3]:", k_full[:, :, 1, :3])

        # ── 6) Split into batch vs. incremental attention paths ──────────
        if kv_cache is None:

            # — Batch mode: attend to all T queries in one shot —
            q = q_all.contiguous()      # (B,H,T,Dh)
            k = k_full.contiguous()     # (B,H,T,Dh)
            v = v_full.contiguous()     # (B,H,T,Dh)

            with open(batch_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["q_raw", str(q[0, 0, :, :].tolist())])  # log head=0 slice for readability
                writer.writerow(["k_raw", str(k[0, 0, :, :].tolist())])
                writer.writerow(["v_raw", str(v[0, 0, :, :].tolist())])

            # ── ▣ DEBUG #3: Compute raw attention scores for last query ▣ ──
            inv_sqrt = 1.0 / math.sqrt(Dh)
            scores_full = torch.einsum("b h q d, b h k d -> b h q k", q, k) * inv_sqrt
            scores_ref_last = scores_full[:, :, -1:, :]  # shape (B,H,1,T)

            # Convert tensors to Python lists or strings for CSV
            scores_last_list = scores_ref_last[0, 0, 0, :].tolist()
            shape_q = tuple(q.shape)
            shape_k = tuple(k.shape)
            shape_v = tuple(v.shape)

            # Append batch‐mode debug values to CSV
            with open(batch_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["scores_ref_last", str(scores_last_list)])
                writer.writerow(["q.shape", str(shape_q)])
                writer.writerow(["k.shape", str(shape_k)])
                writer.writerow(["v.shape", str(shape_v)])

            probs_full = F.softmax(scores_ref_last, dim=-1)  # still (B,H,1,T)
            probs_full_list = probs_full[0, 0, 0, :].tolist()
            with open(batch_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["probs_ref_last (batch softmax)", str(probs_full_list)])

            # print("[DEBUG SCORES-BATCH] head0 scores:", scores_last_list)
            # print(f"[DEBUG SHAPES-BATCH] q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")

            out_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
            # print(f"[DEBUG ATTN-BATCH] out_attn_batch.shape={out_attn.shape}")
            # out_attn: (B, H, T, Dh)

            # ── ▣ DEBUG #4: Print attention output for last query before o_proj ▣ ──
            attn_ref_last = out_attn[:, :, -1:, :]  # (B,H,1,Dh)

            attn_out_vec_batch = attn_ref_last[0, 0, 0, :].tolist()
            with open(batch_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["attn_output_last_query_batch_fullDh", str(attn_out_vec_batch)])

            attn_last3 = attn_ref_last[0, 0, 0, :3].tolist()

            # Log attn output shape and first 3 dims
            shape_out_attn = tuple(out_attn.shape)
            with open(batch_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["out_attn_batch.shape", str(shape_out_attn)])
                writer.writerow(["attn_ref_last_first3", str(attn_last3)])

            # print(f"[DEBUG ATTN-BATCH] out_attn_batch.shape={out_attn.shape}")
            # print("[DEBUG ATTN-BATCH] head0 first3 dims:", attn_last3)

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

            # Ensure CSV exists; if not, write header

            with open(inc_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["q_raw", str(q[0, 0, 0, :].tolist())])  # head=0, only one time‐step
                writer.writerow(["k_raw", str(k[0, 0, :, :].tolist())])  # head=0, all cached+new
                writer.writerow(["v_raw", str(v[0, 0, :, :].tolist())])

            # ── ▣ DEBUG #3b: Print raw attention scores (incremental) ▣ ──
            inv_sqrt = 1.0 / math.sqrt(Dh)
            scores_inc = torch.einsum("b h q d, b h k d -> b h q k", q, k) * inv_sqrt

            # Convert to Python lists/strings for CSV
            scores_inc_list = scores_inc[0, 0, 0, :].tolist()
            shape_q_inc = tuple(q.shape)
            shape_k_inc = tuple(k.shape)
            shape_v_inc = tuple(v.shape)

            # Append incremental‐mode debug values to CSV
            with open(inc_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["scores_inc", str(scores_inc_list)])
                writer.writerow(["q.shape", str(shape_q_inc)])
                writer.writerow(["k.shape", str(shape_k_inc)])
                writer.writerow(["v.shape", str(shape_v_inc)])

            probs_inc = F.softmax(scores_inc, dim=-1)  # shape (B,H,1,T_full)

            # Convert to Python list for CSV
            probs_inc_list = probs_inc[0, 0, 0, :].tolist()

            with open(inc_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["probs_inc (softmaxed scores)", str(probs_inc_list)])

            # print("[DEBUG SCORES-INC]   head0 scores:", scores_inc_list)
            # print(f"[DEBUG SHAPES-INC]   q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")

            masked_inc = scores_inc.clone()  # (B, H, 1, T_full)

            # 2) Softmax along the “k” dimension (dim = -1)
            #    → probs_inc_manual: shape (B, H, 1, T_full)
            probs_inc_manual = torch.softmax(masked_inc, dim=-1)

            attn_inc_manual = torch.einsum(
                "b h q k, b h k d -> b h q d",
                probs_inc_manual,
                v
            )  # shape = (B, H, 1, Dh)

            # Log the manual output vector (full Dh) for debugging
            attn_inc_fullDh = attn_inc_manual[0, 0, 0, :].tolist()
            with open(inc_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["attn_output_last_query_inc_manual_fullDh", str(attn_inc_fullDh)])

            # Call fused SDPA
            # out_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
            out_attn = attn_inc_manual
            # out_attn: (B, H, 1, Dh)

            # ── ▣ DEBUG #4b: Print attention output (incremental) ▣ ──
            attn_inc = out_attn  # (B, H, 1, Dh)

            attn_inc_first3 = attn_inc[0, 0, 0, :3].tolist()
            shape_out_attn_inc = tuple(out_attn.shape)

            attn_inc_fullDh = attn_inc[0, 0, 0, :].tolist()  # length Dh
            with open(inc_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["attn_output_last_query_inc_fullDh", str(attn_inc_fullDh)])
                writer.writerow(["manual_attn_output", str(torch.matmul(probs_inc.unsqueeze(0), v).squeeze(0))])

            with open(inc_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["out_attn_inc.shape", str(shape_out_attn_inc)])
                writer.writerow(["attn_inc_first3", str(attn_inc_first3)])

            # print(f"[DEBUG ATTN-INC]   out_attn_inc.shape={out_attn.shape}")
            # print("[DEBUG ATTN-INC]   head0 first3 dims:", attn_inc_first3)

            # Reshape & project → (B,1,H*Dh) → (B,1,D)
            out = (
                out_attn
                .transpose(1, 2)       # (B, 1, H, Dh)
                .contiguous()
                .view(B, 1, H * Dh)
                .contiguous()
            )
            out = self.o_proj(out)    # (B, 1, D)

            return out, new_cache

def reset_debug_csvs():
    global batch_csv, inc_csv

    for fname in [batch_csv, inc_csv]:
        # Overwrite existing file or create a new one with only the header
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["var_name", "value"])

def test_mha_kv_cache():
    global batch_csv, inc_csv

    reset_debug_csvs()

    torch.manual_seed(0)

    # ----- tiny dummy config -----
    cfg = Config.from_json("../pretrained_1.1b/config.json")

    mha = MHA(cfg)
    mha.eval()

    B, L = 1, 3
    x = torch.randn(B, L, cfg.d_model)

    H, Dh = cfg.n_heads, mha.d_head
    repeat = cfg.n_heads // cfg.n_kv_heads

    # ---------------- helpers -----------------
    def make_q(module, token_slice, offset):
        q = module.q_proj(token_slice.contiguous()) \
            .view(B, 1, H, Dh).transpose(1, 2)
        sin, cos = module._build_rope(offset + 1, token_slice.device)
        sin, cos = sin[:, :, offset:offset + 1, :], cos[:, :, offset:offset + 1, :]
        return apply_rope(q, sin, cos)  # (B,H,1,Dh)

    def expand_kv(t):
        return t.contiguous().repeat_interleave(repeat, dim=1)

    # ------------- reference tensors (tokens 0-1 together) ----------------
    with torch.no_grad():
        # K/V for two-token prefix
        _, ref_cache = mha(x[:, :2], kv_cache=None)

    q_ref = make_q(mha, x[:, 1:2], offset=1)  # absolute position 1
    k_ref = expand_kv(ref_cache["k"])
    v_ref = expand_kv(ref_cache["v"])

    # ------------- incremental tensors (token 0 then token 1) -------------
    cache = None
    _, cache = mha(x[:, :1], kv_cache=None)  # token-0 primes cache
    _, cache = mha(x[:, 1:2], kv_cache=cache)

    q_inc = make_q(mha, x[:, 1:2], offset=cache["k"].size(2) - 1)  # offset = 1
    k_inc = expand_kv(cache["k"])
    v_inc = expand_kv(cache["v"])

    # ------------- compare and LOG differences ----------------------------
    def maxdiff(a, b):
        return (a - b).abs().max().item()

    print("max |Δq| :", maxdiff(q_ref, q_inc))
    print("max |Δk| :", maxdiff(k_ref, k_inc))
    print("max |Δv| :", maxdiff(v_ref, v_inc))

    # ──────────────────────────────────────────────────────────────────────
    # 📒  EXTRA LOG FOR TOKEN-2   (add *after* the token-1 log you already have)
    # ──────────────────────────────────────────────────────────────────────
    cache = None
    pieces_inc = []
    for t in range(3):  # tokens 0,1,2
        out, cache = mha(x[:, t:t + 1], kv_cache=cache)
        pieces_inc.append(out)  # keep outputs
        if t == 2:  # <-- token-2 probe
            k_inc_t2 = cache["k"].contiguous()
            v_inc_t2 = cache["v"].contiguous()
            q_inc_t2 = mha.q_proj(x[:, 2:3]).view(B, 1, H, Dh).transpose(1, 2)
            sin, cos = mha._build_rope(cache["k"].size(2), x.device)
            q_inc_t2 = apply_rope(q_inc_t2, sin[:, :, -1:], cos[:, :, -1:])

    with torch.no_grad():
        _, ref_cache_t2 = mha(x[:, :3], kv_cache=None)  # reference up to token-2

    q_ref_t2 = mha.q_proj(x[:, 2:3]).view(B, 1, H, Dh).transpose(1, 2)
    sin3, cos3 = mha._build_rope(3, x.device)
    q_ref_t2 = apply_rope(q_ref_t2, sin3[:, :, 2:3], cos3[:, :, 2:3])

    k_ref_t2 = ref_cache_t2["k"].contiguous()
    v_ref_t2 = ref_cache_t2["v"].contiguous()

    def md(a, b):
        return (a - b).abs().max().item()

    print("token-2   Δq:", md(q_ref_t2, q_inc_t2))
    print("token-2   Δk:", md(k_ref_t2, k_inc_t2))
    print("token-2   Δv:", md(v_ref_t2, v_inc_t2))
    #
    # ------------------------------------------------------------------
    # Correct equality test: compare ONLY the newest token each step
    # ------------------------------------------------------------------
    cache_inc = None
    for i, t in enumerate(range(L)):
        print("Index", i, "token", t)

        # ── Reference path ───────────────────────────────
        out_ref, ref_cache = mha(x[:, : t + 1], kv_cache=None)
        last_ref = out_ref[:, -1:]  # (B,1,D)

        if t == 1:
            # ------------------- LOG Q DIFFERENCE -------------------
            # Build q_ref exactly as MHA.forward does
            q_ref = mha.q_proj(x[:, 1:2])  # (B, 1, H*Dh)
            q_ref = q_ref.view(B, 1, cfg.n_heads, mha.d_head).transpose(1, 2)  # → (B, H, 1, Dh)

            # Fetch absolute-pos RoPE slice for the second token in a 2-token sequence:
            sin_all_ref, cos_all_ref = mha._build_rope(2, x.device)  # full length=2
            sin_ref = sin_all_ref[:, :, 1:2, :]  # (1,1,1,Dh)
            cos_ref = cos_all_ref[:, :, 1:2, :]

            # Apply RoPE
            q_ref = apply_rope(q_ref, sin_ref, cos_ref)  # (B, H, 1, Dh)

        # ── Incremental path ────────────────────────────
        out_inc, cache_inc = mha(x[:, t: t + 1], kv_cache=cache_inc)
        last_inc = out_inc  # (B,1,D)

        if t == 1:
            # ------------------- LOG K/V DIFFERENCE -------------------
            # Grab raw K/V from the reference cache (before expansion)
            k_ref_raw = ref_cache["k"]       # shape (B, HK, 2, Dh)
            v_ref_raw = ref_cache["v"]       # shape (B, HK, 2, Dh)
            k_ref_full = k_ref_raw.contiguous().repeat_interleave(repeat, dim=1)
            v_ref_full = v_ref_raw.contiguous().repeat_interleave(repeat, dim=1)

            # Grab raw K/V from the incremental cache (before expansion)
            k_inc_raw = cache_inc["k"]       # shape (B, HK, 2, Dh)
            v_inc_raw = cache_inc["v"]       # shape (B, HK, 2, Dh)
            k_inc_full = k_inc_raw.contiguous().repeat_interleave(repeat, dim=1)
            v_inc_full = v_inc_raw.contiguous().repeat_interleave(repeat, dim=1)

            print("  → max|k_ref_full – k_inc_full| =",
                  (k_ref_full - k_inc_full).abs().max().item())
            print("  → max|v_ref_full – v_inc_full| =",
                  (v_ref_full - v_inc_full).abs().max().item())

            # ------------------- LOG Q DIFFERENCE -------------------
            # Build q_inc exactly as MHA.forward does
            q_inc = mha.q_proj(x[:, 1:2])  # (B, 1, H*Dh)
            q_inc = q_inc.view(B, 1, cfg.n_heads, mha.d_head).transpose(1, 2)  # → (B, H, 1, Dh)

            offset_inc = cache_inc["k"].size(2) - 1  # should equal 1 here
            sin_all_inc, cos_all_inc = mha._build_rope(offset_inc + 1, x.device)
            sin_inc = sin_all_inc[:, :, offset_inc: offset_inc + 1, :]  # (1,1,1,Dh)
            cos_inc = cos_all_inc[:, :, offset_inc: offset_inc + 1, :]

            q_inc = apply_rope(q_inc, sin_inc, cos_inc)  # (B, H, 1, Dh)

            diff_q = (q_ref - q_inc).abs().max().item()
            print(f"  → max|q_ref – q_inc| = {diff_q:.6f}")

            # ------------------- LOG ATTENTION INTERNALS -------------------
            inv_sqrt = 1.0 / math.sqrt(mha.d_head)

            # Pre‐softmax scores
            scores_ref = torch.einsum("b h q d, b h k d -> b h q k", q_ref, k_ref_full) * inv_sqrt
            scores_inc = torch.einsum("b h q d, b h k d -> b h q k", q_inc, k_inc_full) * inv_sqrt

            # Softmax probabilities
            probs_ref = torch.softmax(scores_ref, dim=-1)   # (B, H, 1, 2)
            probs_inc = torch.softmax(scores_inc, dim=-1)

            # Weighted sums
            attn_out_ref = torch.einsum("b h q k, b h k d -> b h q d", probs_ref, v_ref_full)
            attn_out_inc = torch.einsum("b h q k, b h k d -> b h q d", probs_inc, v_inc_full)

            # Flatten / transpose to feed o_proj
            out_ref_head = (
                attn_out_ref.transpose(1, 2)                # (B, 1, H, Dh)
                .contiguous()
                .view(B, 1, -1)                              # (B, 1, H*Dh)
            )
            out_inc_head = (
                attn_out_inc.transpose(1, 2)                # (B, 1, H, Dh)
                .contiguous()
                .view(B, 1, -1)
            )

            # Final linear projection
            o_ref = mha.o_proj(out_ref_head)
            o_inc = mha.o_proj(out_inc_head)

            print("  → max|scores_ref – scores_inc| =",
                  (scores_ref - scores_inc).abs().max().item())
            print("  → max|probs_ref – probs_inc| =",
                  (probs_ref - probs_inc).abs().max().item())
            print("  → max|attn_out_ref – attn_out_inc| =",
                  (attn_out_ref - attn_out_inc).abs().max().item())
            print("  → max|o_ref – o_inc| =",
                  (o_ref - o_inc).abs().max().item())

            print("  → max|last_ref - o_ref|  =", (last_ref - o_ref).abs().max().item())
            print("  → max|last_inc - o_inc| =", (last_inc - o_inc).abs().max().item())

        # ── Now compare final logits ───────────────────
        try:
            torch.testing.assert_close(last_ref, last_inc, atol=1e-3, rtol=1e-3)
        except AssertionError:
            print("  !! last_ref vs last_inc mismatch at index", i)
            raise
def test_mha_kv_cache_full_sequence():
    global batch_csv, inc_csv

    """
    Verifies that MHA returns the same last‐token output whether you run:

      (1) Batch mode on a prefix x[:, :t+1] → slice off the last row, or
      (2) Incremental mode one token at a time carrying kv_cache forward.

    For each length L = 1..max_len:
      - Generate a random x of shape (B, L, D).
      - Reference: for each t in [0..L-1], call mha(x[:, :t+1], kv_cache=None),
                   then slice out the last row: out_ref_t = out_ref_full[:, -1:, :].
      - Incremental: feed one token at a time with kv_cache, collect out_inc_t.
      - Compare all stepwise (out_ref_t == out_inc_t) and also the final concatenated.
    """

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


def print_raw_batch():
    torch.manual_seed(0)
    cfg = Config.from_json("../pretrained_1.1b/config.json")
    mha = MHA(cfg)
    mha.eval()

    B = 2
    D = cfg.d_model

    # Only test L=2 for now
    L = 5
    x = torch.randn(B, L, D)

    # 1) Batch on prefix [0,1]
    with torch.no_grad():
        out_ref_full, _ = mha(x, kv_cache=None)  # shape (B, 2, D)
    # Take last token (for token index 1)
    out_ref_last = out_ref_full[:, -1:, :]       # (B, 1, D)

    # 2) Incremental: token‐0, then token‐1
    cache_inc = None
    with torch.no_grad():
        out_inc_0, cache_inc = mha(x[:, 0:1], kv_cache=None)
        out_inc_1, cache_inc = mha(x[:, 1:2], kv_cache=cache_inc)
    # out_inc_1 has shape (B, 1, D)

    print("=== L=2: Batch ‘last‐token’ output ===")
    print(out_ref_last)           # (B, 1, D)
    print("=== L=2: Incremental ‘last‐token’ output ===")
    print(out_inc_1)             # (B, 1, D)

    # Now do a quick max‐difference check:
    diff = (out_ref_last - out_inc_1).abs().max().item()
    print(f"max |out_ref_last - out_inc_1| = {diff:.6f}")

    # Then assert to force the failure again:
    torch.testing.assert_close(out_ref_last, out_inc_1, atol=1e-4, rtol=1e-4)

if __name__ == "__main__":
    test_mha_kv_cache_full_sequence()
    # print_raw_batch()
