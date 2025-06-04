import torch
import torch.nn as nn
from .config import Config
from .transformer import Block
from .rmsnorm import RMSNorm
# from config import Config
# from transformer import Block
# from rmsnorm import RMSNorm

class TinyLlama(nn.Module):
    """
    TinyLlama: embedding → N transformer blocks (with KV‐cache support) → norm → lm_head.

    The forward method accepts:
      - idx:         token indices, shape (B, T) in batch mode or (B, 1) in incremental mode.
      - kv_caches:   None (for batch) or a list of length n_layers of cache dicts (one per block)
                     when decoding incrementally.

    Returns:
      - logits:      shape (B, T, vocab_size) in batch mode; (B, 1, vocab_size) when T=1 in incremental mode.
      - new_caches:  list of length n_layers, each a dict {"k": <...>, "v": <...>} for next incremental step.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.config = cfg

        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    @classmethod
    def from_config(cls, path):
        return cls(Config.from_json(path))

    def forward(
        self,
        input_ids: torch.Tensor,                  # (B, T) or (B, 1)
        kv_caches: [list[dict] , None] = None  # either None or list of length n_layers
    ) -> tuple[torch.Tensor, list[dict]]:
        B, T = input_ids.shape

        # 1) Token embedding
        x = self.emb(input_ids)  # (B, T, D)

        new_caches = []

        # 2) Pass through each block, carrying kv_cache per block if provided
        for i, blk in enumerate(self.blocks):
            cache_i = None if kv_caches is None else kv_caches[i]
            x, new_cache_i = blk(x, kv_cache=cache_i)
            # x: (B, T, D) in batch mode; or (B, 1, D) in incremental mode
            new_caches.append(new_cache_i)

        # 3) Final RMSNorm + lm_head
        x = self.norm(x)            # (B, T, D) or (B, 1, D)
        logits = self.lm_head(x)     # (B, T, V) or (B, 1, V)

        return logits, new_caches

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: list[dict] = None,
            attention_mask: torch.LongTensor = None,
            **kwargs
    ):
        """
        This method is used by PEFT (and HF generation) to package up inputs for the next token.
        For a “decoder‐only” model like TinyLlama, you simply return a dict of kwargs that forward() expects.
        """

        # If past_key_values is not None, we are in “incremental” mode;
        # the next token to feed is the last token of input_ids:
        if past_key_values is not None:
            # Only keep the last token in incremental decoding
            input_ids = input_ids[:, -1:].contiguous()

        return {
            "idx": input_ids,  # matches your forward signature
            "kv_caches": past_key_values
            # (you could pass attention_mask here if you had one)
        }

def test_tinyllama_kv_cache_full_sequence():
    """
    Verifies that TinyLlama produces identical logits when run:
      1) In “batch” mode on prefixes [0..t] (no caches), versus
      2) In “incremental” mode one token at a time carrying caches forward.

    For each sequence length L=1..max_len:
      - Create random token indices idx of shape (B, L).
      - Reference: for each t in [0..L-1], call model(idx[:, :t+1], kv_caches=None) to get logits_ref_t (B, 1, V).
      - Incremental: initialize kv_caches = [None]*n_layers, then for t in 0..L-1:
          logits_inc_t, kv_caches = model(idx[:, t:t+1], kv_caches)
      - Compare stepwise logits (shape (B,1,V)) and final concatenated (B, L, V).
    """

    torch.manual_seed(0)

    cfg = Config.from_json("../pretrained_1.1b/config.json")
    model = TinyLlama(cfg)
    model.eval()

    B = 5
    V = cfg.vocab_size
    max_len = 20

    for L in range(1, max_len + 1):
        # --- 1) Random token indices of shape (B, L) ---
        idx = torch.randint(0, V, (B, L), dtype=torch.long)

        # --- 2) Reference: batch on each prefix, no caches ---
        reference_logits = []
        for t in range(L):
            idx_prefix = idx[:, :t + 1]  # shape (B, t+1)
            with torch.no_grad():
                logits_ref_full, _ = model(idx_prefix, kv_caches=None)
                # logits_ref_full has shape (B, t+1, V)
            logits_ref_last = logits_ref_full[:, -1:]  # (B, 1, V)
            reference_logits.append(logits_ref_last)

        # Concatenate to get (B, L, V)
        full_ref_cat = torch.cat(reference_logits, dim=1)

        # --- 3) Incremental: feed one token at a time, carrying kv_caches ---
        kv_caches = [None] * cfg.n_layers
        incremental_logits = []

        for t in range(L):
            token_id = idx[:, t : t + 1]  # (B, 1)
            with torch.no_grad():
                logits_inc_t, kv_caches = model(token_id, kv_caches)
                # logits_inc_t has shape (B, 1, V)
            incremental_logits.append(logits_inc_t)

            # Stepwise check: token‐t logits should match reference_logits[t]
            torch.testing.assert_close(
                reference_logits[t],
                logits_inc_t,
                atol=1e-4,
                rtol=1e-4,
            )

        # Concatenate incremental logits to (B, L, V)
        incremental_cat = torch.cat(incremental_logits, dim=1)

        # Final check: entire sequence logits should match
        torch.testing.assert_close(
            full_ref_cat,
            incremental_cat,
            atol=1e-5,
            rtol=1e-5,
        )

        print(f"  ✓ L={L} passed (TinyLlama batch vs. incremental match)")

    print("✓ test_tinyllama_kv_cache_full_sequence passed for lengths 1..", max_len)


if __name__ == "__main__":
    test_tinyllama_kv_cache_full_sequence()