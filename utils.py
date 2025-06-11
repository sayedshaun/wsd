# Copyright (c) 2025, Sayed Shaun.  All rights reserved.

import os
import torch
import random
import numpy as np
from typing import List, Tuple, Union
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


def seed_everything(seed: Union[int, None] = None) -> None:
    """
    Set random seed for reproducibility.
    """
    if seed is None:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_model_weights(model: torch.nn.Module, weight_dir: str, device: str) -> None:
    """
    Load model weights from the specified directory.
    """
    weight_list = os.listdir(weight_dir)
    path = sorted(weight_list, key=lambda x: x.split("-")[-1])[-1]
    print("Loading model from: ", path)
    state_dict = torch.load(os.path.join(weight_dir, path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_exact_match(pred_span: Tuple[int, int], gold_span: Tuple[int, int]) -> int:
    return int(pred_span == gold_span)

def compute_f1_span(pred_span: Tuple[int, int], gold_span: Tuple[int, int]) -> float:
    pred_set = set(range(pred_span[0], pred_span[1] + 1))
    gold_set = set(range(gold_span[0], gold_span[1] + 1))
    if not pred_set or not gold_set:
        return 0.0
    overlap = pred_set & gold_set
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred_set)
    recall = len(overlap) / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def eval_metrics_for_span_extraction(
    pred_start: List[int], 
    pred_end: List[int], 
    start_positions: List[int],
    end_positions: List[int]
) -> Tuple[float, float]:
    """
    Evaluate span extraction with span-level EM and F1.
    """
    exact_matches = []
    f1_scores = []

    for ps, pe, gs, ge in zip(pred_start, pred_end, start_positions, end_positions):
        pred_span = (ps, pe)
        gold_span = (gs, ge)
        exact_matches.append(compute_exact_match(pred_span, gold_span))
        f1_scores.append(compute_f1_span(pred_span, gold_span))
    
    em = np.mean(exact_matches)
    f1 = np.mean(f1_scores)

    return round(em * 100, 2), round(f1 * 100, 2)



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
                    device: str) -> Tuple[float, float, float]:
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
        em, f1 = eval_metrics_for_span_extraction(
            pred_start=start_pred, 
            pred_end=end_pred, 
            start_positions=start_true, 
            end_positions=end_true
        )
        start_true, start_pred = [], []
        end_true, end_pred = [], []
        model.train()
        return loss, em, f1