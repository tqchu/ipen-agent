import torch
import os
from llm.tinyllama.core.model import TinyLlama
from llm.tinyllama.io.hf_state import load_state
from llm.tinyllama.io.tokenizer import Tok


def initialize_model():
    """Initialize the model and tokenizer once to avoid reloading"""

    # Get the directory where this file (loader.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Set model directory relative to this file's location
    model_dir = os.path.join(current_dir, "pretrained_1.1b")

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer with absolute path
    tokenizer_path = os.path.join(model_dir, "tokenizer.model")
    tokenizer = Tok(tokenizer_path)

    # Load model config with absolute path
    config_path = os.path.join(model_dir, "config.json")
    model = TinyLlama.from_config(config_path).to(device)

    # Load state dict
    with torch.no_grad():
        model.load_state_dict(load_state(model_dir), strict=True)

    return tokenizer, model
