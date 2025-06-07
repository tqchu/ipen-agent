import time

import torch

from llm.tinyllama.core.model import TinyLlama
from llm.tinyllama.io.tokenizer import format_chat_prompt, Tok



def get_answer(system_prompt: str, question: str, model:TinyLlama, tokenizer, max_tokens, no_cache = False) -> (str, float):
    """Get an answer from the model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prompt = format_chat_prompt(
        tokenizer, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ], add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer.encode(prompt)

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    generated = []

    model.eval()
    with torch.no_grad():
        logits_full, kv_caches = model(input_tensor, kv_caches=None)

        # Grab the last‐token logits from the prompt
        next_logits = logits_full[0, -1, :]  # shape (vocab_size,)

        # 3. Generate tokens one by one
        for _ in range(max_tokens):
            # 3a. Top‐k sampling from the current logits
            # probs = next_logits.softmax(dim=-1)  # (vocab_size,)
            # topk_probs, topk_idxs = torch.topk(probs, k=50)
            # topk_probs = topk_probs / topk_probs.sum()  # renormalize
            # pick = torch.multinomial(topk_probs, num_samples=1).item()
            # token_id = topk_idxs[pick].item()
            token_id = next_logits.argmax(dim=-1).item()

            # If EOS, stop generation
            if token_id == tokenizer.eos_id:
                break

            generated.append(token_id)

            if no_cache:
                input_tensor = torch.cat([input_tensor, torch.tensor([[token_id]], device=device)], dim=1)
                with torch.no_grad():
                    logits_step, kv_caches = model(input_tensor, kv_caches=None)
            else:
                new_input = torch.tensor([[token_id]], device=device)  # (1, 1)
                with torch.no_grad():
                    logits_step, kv_caches = model(new_input, kv_caches=kv_caches)

            next_logits = logits_step[0, -1, :]  # shape (vocab_size,)

    # 4. Decode the generated tokens (excluding the prompt)
    generated_text = tokenizer.decode(generated)

    # 5. Dummy confidence
    conf = 1.0

    return generated_text, conf


