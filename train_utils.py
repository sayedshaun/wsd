import os
from typing import Tuple
import wandb
import torch
from tqdm import tqdm
from sklearn import metrics
from transformers import get_linear_schedule_with_warmup
from utils import eval_metrics_for_span_extraction




def train_fn(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 1e-5,
        logging_step: int = 1000,
        precision: torch.dtype = torch.float16,
        warmup_ratio: float = 0.1,
        grad_clip: float = 1.0,
        weight_decay: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'output',
        report_to: str = None
    ) -> None:
    if report_to == 'wandb':
        wandb.init(project='WordSenseDisambiguation', name=output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=int(warmup_ratio * total_steps),
                    num_training_steps=total_steps)
    scaler = torch.amp.GradScaler()
    global_step = 1
    best_f1 = float('-inf')
    with tqdm(total=total_steps, desc="Training") as pbar:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            train_y_true, train_y_pred = [], []

            for batch in train_dataloader:
                optimizer.zero_grad()
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor)
                    else [g.to(device) for g in v]) for k, v in batch.items()
                }
                with torch.amp.autocast_mode.autocast(
                    device_type=device,
                    enabled=torch.cuda.is_available(), 
                    dtype=precision
                    ):
                    outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_y_true.extend(batch['labels'].cpu().numpy())
                train_y_pred.extend(outputs.logits.argmax(dim=1).cpu().numpy())
                if global_step % logging_step == 0 and global_step > 0:
                    train_accu = metrics.accuracy_score(train_y_true, train_y_pred)
                    t_p, t_r, t_f1, t_s = metrics.precision_recall_fscore_support(
                        train_y_true, train_y_pred, average='macro',zero_division=0)
                    train_y_true, train_y_pred = [], []

                    v_l, v_f1, v_p, v_r, v_a = evaluation_fn(model, val_dataloader, device)
                    if report_to == 'wandb':
                        wandb.log(
                            {
                                'train/epoch': epoch,
                                'train/global_step': global_step,
                                'train/loss': total_loss / logging_step,
                                'train/accuracy': train_accu,
                                'train/f1': t_f1,
                                'train/precision': t_p,
                                'train/recall': t_r,
                                'validation/loss': v_l,
                                'validation/precision': v_p,
                                'validation/recall': v_r,
                                'validation/accuracy': v_a,
                                'validation/f1': v_f1,
                                'validation/best_f1': best_f1
                            }
                        )
                    total_loss = 0.0

                    # Save checkpoints
                    if v_f1 > best_f1:
                        best_f1 = v_f1
                        torch.save(model.state_dict(), os.path.join(output_dir, f"step-{global_step}-f1-{v_f1}.pt"))
                        
                        # Sorted by F1 score and keep 3 checkpoints
                        checkpoint_files = [f for f in os.listdir(output_dir) if f.endswith('.pt') and '-' in f]
                        checkpoints = sorted(
                            checkpoint_files,
                            key=lambda x: float(x.split('-')[3].replace(".pt", "")),
                            reverse=True
                        )
                        for ckpt in checkpoints[3:]:
                            os.remove(os.path.join(output_dir, ckpt))

                global_step += 1
                pbar.update(1)
    if report_to == 'wandb':
        wandb.finish()




def evaluation_fn(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                            device: str) -> Tuple[float, float, float, float, float]:
    with torch.no_grad():
        y_true, y_pred = [], []
        model.eval()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc="Evaluating"):
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



def span_train_fn(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 1e-5,
        logging_step: int = 1000,
        precision: torch.dtype = torch.float16,
        warmup_ratio: float = 0.1,
        grad_clip: float = 1.0,
        weight_decay: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'output',
        report_to: str = None
    ) -> None:
    if report_to == 'wandb':
        wandb.init(project='WordSenseDisambiguation', name=output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=int(warmup_ratio * total_steps),
                    num_training_steps=total_steps)
    scaler = torch.amp.GradScaler()
    global_step = 1
    best_accuracy = float('-inf')
    with tqdm(total=total_steps, desc="Training") as pbar:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            train_start_true, train_start_pred = [], []
            train_end_true, train_end_pred = [], []

            for batch in train_dataloader:
                optimizer.zero_grad()
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor)
                    else [g.to(device) for g in v]) for k, v in batch.items()
                }
                with torch.amp.autocast_mode.autocast(
                    device_type=device,
                    enabled=torch.cuda.is_available(), 
                    dtype=precision
                    ):
                    outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_start_true.extend(batch['start_positions'].cpu().numpy())
                train_start_pred.extend(outputs.start_logits.argmax(dim=1).cpu().numpy())
                train_end_true.extend(batch['end_positions'].cpu().numpy())
                train_end_pred.extend(outputs.end_logits.argmax(dim=1).cpu().numpy())
                # Logging
                if global_step % logging_step == 0 and global_step > 0:
                    start_accu, end_accu, joint_accu = eval_metrics_for_span_extraction(
                        torch.tensor(train_start_pred), 
                        torch.tensor(train_end_pred), 
                        torch.tensor(train_start_true), 
                        torch.tensor(train_end_true)
                    )
                    train_start_true, train_start_pred = [], []
                    train_end_true, train_end_pred = [], []

                    val_loss, v_start_acc, v_end_acc, v_joint_acc = span_evaluation_fn(model, val_dataloader, device)
                    if report_to == 'wandb':
                        wandb.log(
                            {
                                'train/epoch': epoch,
                                'train/global_step': global_step,
                                'train/loss': total_loss / logging_step,
                                'train/start_accuracy': start_accu,
                                'train/end_accuracy': end_accu,
                                'train/joint_accuracy': joint_accu,
                                'validation/loss': val_loss,
                                'validation/start_accuracy': v_start_acc,
                                'validation/end_accuracy': v_end_acc,
                                'validation/joint_accuracy': v_joint_acc,
                                'validation/best_accuracy': best_accuracy
                            }
                        )
                    total_loss = 0.0

                    # Save checkpoints
                    if v_joint_acc > best_accuracy:
                        best_accuracy = v_joint_acc
                        torch.save(model.state_dict(), os.path.join(output_dir, f"step-{global_step}-f1-{best_accuracy}.pt"))
                        
                        # Sorted by F1 score and keep 3 checkpoints
                        checkpoint_files = [f for f in os.listdir(output_dir) if f.endswith('.pt') and '-' in f]
                        checkpoints = sorted(
                            checkpoint_files,
                            key=lambda x: float(x.split('-')[3].replace(".pt", "")),
                            reverse=True
                        )
                        for ckpt in checkpoints[3:]:
                            os.remove(os.path.join(output_dir, ckpt))

                global_step += 1
                pbar.update(1)
    if report_to == 'wandb':
        wandb.finish()



def span_evaluation_fn(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                    device: str) -> Tuple[float, float, float, float, float]:
    with torch.no_grad():
        start_true, start_pred = [], []
        end_true, end_pred = [], []
        model.eval()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc="Evaluating"):
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
            torch.tensor(start_pred), 
            torch.tensor(end_pred), 
            torch.tensor(start_true), 
            torch.tensor(end_true)
        )
        print(f"Start Accuracy: {start_acc}, End Accuracy: {end_acc}, Joint Accuracy: {joint_acc}")
        start_true, start_pred = [], []
        end_true, end_pred = [], []
        model.train()
        return loss, start_acc, end_acc, joint_acc