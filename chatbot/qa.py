import logging
import sys
import time
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers import TextStreamer, GenerationConfig
import torch, math, time

from llm import tinyllama
from llm.tinyllama import loader
from llm.tinyllama.qa import q_a

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PenTest-AI")


class ModelManager:
    """Singleton class to manage the model and tokenizer instances."""
    _instance = None

    def __new__(cls, use_huggingface: bool = True):
        if cls._instance is None:
            logger.info("Initializing the AI model...")
            cls._instance = super(ModelManager, cls).__new__(cls)
            if use_huggingface:
                cls._instance.tokenizer, cls._instance.model = initialize_model()
            else:
                cls._instance.tokenizer, cls._instance.model = loader.initialize_model()
            logger.info("AI model initialized successfully")
            cls.use_huggingface = use_huggingface
        return cls._instance

    def get_answer(self, question: str, system_prompt=(
            "You are a helpful penetration-testing assistant.\n"
            "Respond **only with the final answer**, do not reveal your chain of thought."
    ), max_tokens : int = 512) -> (str, float):
        start = time.time()

        """Get an answer from the model."""
        if self.use_huggingface:
            answer, conf = get_answer(
                question,
                system_prompt=system_prompt,
                tokenizer=self.tokenizer,
                model=self.model,
                max_tokens=max_tokens
            )
        else:
            answer, conf = q_a.get_answer(
                system_prompt=system_prompt,
                question=question,
                tokenizer=self.tokenizer,
                model=self.model,
                max_tokens=max_tokens
            )

        logger.info(f"[Answering] Response time: {time.time() - start:.2f} seconds")

        return answer, conf


def initialize_model():
    """Initialize the model and tokenizer once to avoid reloading"""
    # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    # model = AutoModelForCausalLM.from_pretrained(
    #     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype="auto", device_map="auto")
    return tokenizer, model


def get_answer(
        question: str,
        system_prompt: str = "You are a helpful penetration-testing assistant.",
        conf_tokens: int = 32,
        max_tokens: int = 512,
        tokenizer=None,
        model=None,
        stream=False,
        **gen_kwargs,
) -> (str, float):
    if tokenizer is None or model is None:
        tokenizer, model = initialize_model()

    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    streamer = TextStreamer(tokenizer) if stream else None
    cfg = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.8,
        return_dict_in_generate=True,  # ← required
        output_scores=True,
        **gen_kwargs,
    )
    out = model.generate(**inputs, streamer=streamer, generation_config=cfg)

    if stream:  # streamer already printed tokens
        return None

    # --- decode ONLY the tokens beyond the prompt -------------------------
    new_ids = out.sequences[0][input_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    text = text.split("<|im_end|>")[0].strip()  # stop token
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # gen_out.scores[n] = logits for step n (before softmax)
    logprobs = []
    for step, logits in enumerate(out.scores[:conf_tokens]):
        # token generated at this step
        tok_id = new_ids[step]
        lp = log_softmax(logits, dim=-1)[0, tok_id].item()
        logprobs.append(lp)

    # mean log-probability → probability space
    conf = math.exp(sum(logprobs) / len(logprobs)) if logprobs else 0.0

    return text, conf


if __name__ == "__main__":
    # Load model once
    model_manager = ModelManager(use_huggingface=False)

    start = time.time()
    # Test with a sample question
    print("\nTesting the chatbot with a sample question:")
    answer, conf = model_manager.get_answer("What is penetration testing?")
    print(f"Answer with confidence of {conf}: {answer}")

    print(f"Response time: {time.time() - start:.2f} seconds")
    # Interactive mode
    print("\nEnter questions (or 'exit' to quit):")
    while True:
        question = input("\nYour question: ")
        if question.lower() in ['exit', 'quit', 'q']:
            break
        start = time.time()
        answer = model_manager.get_answer(question)
        print(f"Answer with confidence of {conf}: {answer}")
        print(f"Response time: {time.time() - start:.2f} seconds")
