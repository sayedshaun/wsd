import os
import yaml
import torch
from data_builder import DataBuilder
from dataset import WSDDataset
from model import WordSenseDisambiguationModel
from torch.utils.data import DataLoader
import argparse
from utils import get_tokenizer, trainable_params, seed_everything
from train_utils import train_fn


if not os.path.exists("data/Training_Corpora") and not os.path.exists("data/Evaluation_Datasets"):
    os.system("bash download.sh")



def main(args: argparse.Namespace):
    # Some assertions to check if the arguments are valid
    assert args.epochs > 0, "Epochs should be greater than 0"
    assert args.max_seq_len > 0, "Max sequence length should be greater than 0"
    assert args.lr > 0, "Learning rate should be greater than 0"
    assert args.batch_size > 0, "Batch size should be greater than 0"
    assert args.num_sense > 0, "Number of sense should be greater than 0"
    assert args.logging_step > 0, "Logging step should be greater than 0"
    assert args.seed > 0, "Seed should be greater than 0"
    assert args.precision in ["fp16", "fp32", "bf16"], "Precision should be fp16, fp32 or bf16"
    assert args.device in ["cuda", "cpu"], "Device should be cuda or cpu"

    seed_everything(args.seed)
    tokenizer = get_tokenizer(args.model_name)
    train_dataset = WSDDataset(
        dataframe=DataBuilder(args.train_data_dir, args.pos_tag).to_pandas(), 
        tokenizer=tokenizer, 
        n_sense=args.num_sense, 
        max_seq_length=args.max_seq_len
    )
    val_dataset = WSDDataset(
        dataframe=DataBuilder(args.val_data_dir, args.pos_tag).to_pandas(), 
        tokenizer=tokenizer, n_sense=args.num_sense, max_seq_length=args.max_seq_len
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, 
        pin_memory=(args.device == "cuda"), num_workers=args.workers
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False, 
        pin_memory=(args.device == "cuda"), num_workers= args.workers
    )
    model = WordSenseDisambiguationModel(model_name=args.model_name, tokenizer=tokenizer)
    print(f"Trainable params: {trainable_params(model)}")

    if args.precision == "fp16": args.precision = torch.float16
    if args.precision == "fp32": args.precision = torch.float32
    if args.precision == "bf16": args.precision = torch.bfloat16

    train_fn(
        model=model, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        epochs=args.epochs, 
        output_dir=args.output_dir,
        report_to=args.report_to,
        lr=args.lr,
        weight_decay=args.weight_decay,
        logging_step=args.logging_step,
        precision=args.precision,
        device=args.device,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip
    )

parser = argparse.ArgumentParser(description="Word Sense Disambiguation")
parser.add_argument("-c", type=str, required=True, help="Config file path")
args = parser.parse_args()
config = yaml.safe_load(open("config.yaml"))
args = argparse.Namespace(**config)


if __name__ == "__main__":
    main(args=args)