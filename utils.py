import torch
import numpy as np
import random
import os
from transformers import AutoTokenizer


def get_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<classify>", "</classify>"]}
    )
    return tokenizer


def trainable_params(model: torch.nn.Module) -> int | float:
    params = 0
    for param in model.parameters():
        if param.requires_grad:
            params += param.numel()
    return f"{params / 1e6:.2f}M"


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True