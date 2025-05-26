import sentencepiece as spm

def format_chat_prompt(tokenizer, messages, tokenize=False, add_generation_prompt=True, return_tensors=None):
    # Build the prompt string according to TinyLlama's chat template
    prompt_str = ""
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unknown role: {role}")
        # Append role tag and content with EOS
        prompt_str += f"<|{role}|>\n{content}{tokenizer.eos_token}"
        if tokenize is False:
            prompt_str += "\n"  # newline for readability (optional)
        # After adding content, if this is the last message and we need a generation prompt:
        if i == len(messages) - 1 and add_generation_prompt:
            prompt_str += f"\n<|assistant|>"
    if tokenize:
        # Encode the prompt string to token IDs (adding BOS/EOS as configured in the tokenizer)
        # add_special_tokens=True will add BOS at start (and EOS at end if configured).
        encoding = tokenizer(prompt_str, return_tensors=return_tensors, add_special_tokens=True)
        return encoding
    return prompt_str

class Tok:
    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.bos_id = self.sp.bos_id()   # usually 1
        self.eos_id = self.sp.eos_id()   # usually 2
        self.eos_token = '</s>'

    def encode(self, text, add_bos=True, add_eos=False):
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        # drop special tokens so we don't print <s> </s>
        ids = [i for i in ids if i not in (self.bos_id, self.eos_id)]
        return self.sp.decode(ids)
