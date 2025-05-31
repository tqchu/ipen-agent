import torch
from unittest.mock import patch

from llm.tinyllama.core.config import Config
from llm.tinyllama.core.model import TinyLlama
from llm.tinyllama.qa import q_a


# --- MockTokenizer and MockTinyLlama from earlier ---
class MockTokenizer:
    def __init__(self):
        self.vocab = {"A": 0, "B": 1, "C": 2, "EOS": 3}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.eos_id = self.vocab["EOS"]
        self.eos_token = "</s>"

    def encode(self, text: str):
        # Instead of requiring text in our tiny vocab, always return [0].
        # That means ANY prompt string will be encoded as token‐0 (“A”).
        return [0]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self.inv_vocab[i] for i in token_ids)


class MockTinyLlama:
    def __init__(self, vocab_size: int, n_layers: int = 1):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.step = 0
        self.next_sequence = [1, 2, 3]  # B, C, EOS
        self._dummy_cache = [{"k": None, "v": None} for _ in range(n_layers)]

    def forward(self, input_ids: torch.Tensor, kv_caches=None):
        B, T = input_ids.shape
        if self.step < len(self.next_sequence):
            chosen_token = self.next_sequence[self.step]
        else:
            chosen_token = 3  # always EOS if out of steps
        self.step += 1

        if kv_caches is None:
            logits = torch.zeros(B, T, self.vocab_size)
            logits[:, -1, chosen_token] = 10.0
        else:
            logits = torch.zeros(B, 1, self.vocab_size)
            logits[:, 0, chosen_token] = 10.0

        return logits, self._dummy_cache

    __call__ = forward

# --- The test for get_answer(system_prompt, question, ...) ---
def test_get_answer_mock_with_signature():
    """
    Verifies that get_answer(...) (with its signature: system_prompt, question, model, tokenizer, max_tokens)
    produces the expected generated text and confidence, by patching format_chat_prompt to return "A".
    """

    mock_tokenizer = MockTokenizer()
    vocab_size = len(mock_tokenizer.vocab)  # = 4
    mock_model = MockTinyLlama(vocab_size=vocab_size, n_layers=1)

    dummy_system = "You are a test"
    dummy_question = "What is education?"
    max_tokens = 5

    # Directly call get_answer without patching format_chat_prompt

    generated_text, confidence = q_a.get_answer(
        system_prompt=dummy_system,
        question=dummy_question,
        model=mock_model,
        tokenizer=mock_tokenizer,
        max_tokens=max_tokens,
    )

    # As before, the mock model’s next‐tokens are [B, C, EOS]
    assert generated_text == "B C"
    assert confidence == 1.0

    print("✓ test_get_answer_mock_no_patch passed!")


if __name__ == "__main__":
    test_get_answer_mock_with_signature()
