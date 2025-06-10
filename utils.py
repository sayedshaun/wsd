# Copyright (c) 2025, Sayed Shaun.  All rights reserved.

import os
import torch
import random
import numpy as np
from typing import List, Tuple
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


def eval_metrics_for_span_extraction(
        pred_start: List[int], 
        pred_end: List[int], 
        start_positions: List[int],
        end_positions: List[int]
        ) -> Tuple[float, float, float]:
    """
    Calculate evaluation F1 metrics for span extraction tasks.
    """
    start_f1 = metrics.f1_score(
        start_positions, pred_start, average='micro',
        zero_division=0
    )
    end_f1 = metrics.f1_score(    
        end_positions, pred_end, average='micro',
        zero_division=0
    )
    joint_f1 = np.mean((start_f1, end_f1))
    return (round(start_f1, 4), round(end_f1, 4), round(joint_f1, 4))


def evaluation_fn(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                            device: str) -> Tuple[float, float, float, float, float]:
    with torch.no_grad():
        y_true, y_pred = [], []
        model.eval()
        total_loss = 0.0
        for batch in dataloader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor)
                else [g.to(device) for g in v]) for k, v in batch.items()
            }
            outputs = model(**batch)  # (batch, n_sense)
            y_true.extend(batch['labels'].cpu().numpy())
            y_pred.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            loss = outputs.loss.item()
            total_loss += loss

        loss = total_loss / len(dataloader)
        precision, recall, f1, support = metrics.precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0)
        acc = metrics.accuracy_score(y_true, y_pred)
        print(metrics.classification_report(y_true, y_pred, zero_division=0))
        y_true, y_pred = [], []
        model.train()

    return round(loss,4), round(f1,4), round(precision,4), round(recall,4), round(acc,4)


def span_evaluation_fn(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                    device: str) -> Tuple[float, float, float, float, float]:
    with torch.no_grad():
        start_true, start_pred = [], []
        end_true, end_pred = [], []
        model.eval()
        total_loss = 0.0
        for batch in dataloader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor)
                else [g.to(device) for g in v]) for k, v in batch.items()
            }
            outputs = model(**batch)  # (batch, n_sense)
            start_true.extend(batch['start_positions'].cpu().numpy())
            start_pred.extend(outputs.start_logits.argmax(dim=1).cpu().numpy())
            end_true.extend(batch['end_positions'].cpu().numpy())
            end_pred.extend(outputs.end_logits.argmax(dim=1).cpu().numpy())
            total_loss += outputs.loss.item()

        loss = total_loss / len(dataloader)
        start_acc, end_acc, joint_acc = eval_metrics_for_span_extraction(
            pred_start=start_pred, 
            pred_end=end_pred, 
            start_positions=start_true, 
            end_positions=end_true
        )
        start_true, start_pred = [], []
        end_true, end_pred = [], []
        model.train()
        return loss, start_acc, end_acc, joint_acc