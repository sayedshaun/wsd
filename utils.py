import os
import torch
import random
import numpy as np
from typing import Tuple
from sklearn import metrics
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


def load_model_weights(model: torch.nn.Module, weight_dir: str, device: str) -> None:
    weight_list = os.listdir(weight_dir)
    path = sorted(weight_list, key=lambda x: x.split("-")[-1])[-1]
    print("Loading model from: ", path)
    state_dict = torch.load(os.path.join(weight_dir, path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def eval_metrics_for_span_extraction(pred_start: torch.Tensor, 
                                     pred_end: torch.Tensor, 
                                     start_positions: torch.Tensor, 
                                     end_positions: torch.Tensor
                                     ) -> Tuple[float, float, float]:
    start_accuracy = metrics.f1_score(
        start_positions.cpu().numpy(),
        pred_start.cpu().numpy(),
        average="macro",
        zero_division=0
    )
    end_accuracy = metrics.f1_score(
        end_positions.cpu().numpy(),
        pred_end.cpu().numpy(),
        average="macro",
        zero_division=0
    )
    joint_accuracy = np.mean(
        (start_positions.cpu().numpy() == pred_start.cpu().numpy()) &
        (end_positions.cpu().numpy() == pred_end.cpu().numpy())
    )
    return round(start_accuracy, 4), round(end_accuracy, 4), round(joint_accuracy, 4)