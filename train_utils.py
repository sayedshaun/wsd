import os
from typing import Tuple
import wandb
import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score




def train_fn(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 1e-5,
        logging_step: int = 1000,
        precision: torch.dtype = torch.float16,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = 'output',
        report_to: str = None
    ) -> None:
    if report_to == 'wandb':
        wandb.init(project='WordSenseDisambiguation', name=output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=int(0.1 * total_steps),
                    num_training_steps=total_steps)
    global_step = 1
    best_f1 = float('-inf')
    with tqdm(total=total_steps, desc="Training") as pbar:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
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
                loss.backward()
                optimizer.step()
                scheduler.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                total_loss += loss.item()
                if global_step % logging_step == 0 and global_step > 0:
                    l, f1, p, r, a = evaluation_fn(model, val_dataloader, device)
                    if report_to == 'wandb':
                        wandb.log({
                            'train_loss': total_loss / logging_step,
                            'val_loss': l,
                            'val_precision': p,
                            'val_recall': r,
                            'val_accuracy': a,
                            'val_f1': f1,
                        })
                    # Save checkpoints
                    if f1 > best_f1:
                        best_f1 = f1
                        torch.save(model.state_dict(), os.path.join(output_dir, f"step-{global_step}-f1-{f1}.pt"))
                        
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


def evaluation_fn(
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        device: str
    ) -> Tuple[float, float, float, float, float]:
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
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        y_true, y_pred = [], []
        model.train()

    return round(loss,3), round(f1,3), round(precision,3), round(recall,3), round(acc,3)