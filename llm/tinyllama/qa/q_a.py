import time

import torch

from llm.tinyllama.core.model import TinyLlama
from llm.tinyllama.io.tokenizer import format_chat_prompt, Tok


def answer(system_prompt: str, question: str, model: TinyLlama, tokenizer: Tok) -> (str, float):
    """Get an answer from the model."""
    start = time.time()

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
        # logits = model(input_tensor).logits
        logits = model(input_tensor)
        next_logits = logits[0, -1, :]

        for _ in range(1024):
            # top-k sampling
            probs = next_logits.softmax(-1)
            topk_probs, topk_idxs = torch.topk(probs, k=50)
            topk_probs = topk_probs / topk_probs.sum()
            idx = torch.multinomial(topk_probs, num_samples=1).item()
            token = topk_idxs[idx].item()

            if token == tokenizer.eos_id:
                break
            generated.append(token)

            # append & run again
            input_tensor = torch.cat([input_tensor, torch.tensor([[token]], device=device)], dim=1)
            next_logits = model(input_tensor)[0, -1, :]

    # Calculate confidence (dummy implementation)
    conf = 1.0  # Placeholder for actual confidence calculation

    print("Inference time:", time.time() - start)

    return tokenizer.decode(generated,), conf