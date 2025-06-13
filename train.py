import os
import glob
import json
import yaml
import datetime
import wandb
import shutil
import torch
import argparse
import subprocess
from typing import List
from data_builder import DataBuilder
from dataset import WSDDataset, SpanDataset
from model import WSDModel, SpanExtractionModel
from torch.utils.data import DataLoader
from utils import get_tokenizer, seed_everything, plot_line
from train_utils import train_fn, span_train_fn


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
    assert args.precision in ["fp16", "fp32", "bf16"], "Precision should be fp16, fp32 or bf16"
    assert args.device in ["cuda", "cpu"], "Device should be cuda or cpu"
    assert args.architecture in ["span", "cosine"], (
        "Architecture should be `span` or `cosine`"
    )
    if args.seed is None or args.seed == "none":
        args.seed = None
    if args.precision == "fp16": args.precision = torch.float16
    if args.precision == "fp32": args.precision = torch.float32
    if args.precision == "bf16": args.precision = torch.bfloat16
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copyfile(config_file, os.path.join(args.output_dir, "config.yaml"))
    seed_everything(args.seed)
    tokenizer = get_tokenizer(args.model_name)

    if args.report_to == "wandb":
        wandb.init(
            project="wsd",
            name=f"{args.architecture}_{args.model_name}_{datetime.datetime.now().strftime('%d-%m-%Y')}",
            config=vars(args),
            dir=args.output_dir,
        )

    if args.architecture == "span":
        train_dataset = SpanDataset(
            dataframe=DataBuilder(args.train_data_dir, args.pos, args.seed).to_pandas(), 
            tokenizer=tokenizer, max_seq_length=args.max_seq_len
        )
        val_dataset = SpanDataset(
            dataframe=DataBuilder(args.val_data_dir, args.pos, args.seed).to_pandas(), 
            tokenizer=tokenizer, max_seq_length=args.max_seq_len
        )
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True, 
            pin_memory=(args.device == "cuda"), num_workers=args.workers
        )
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=args.batch_size, shuffle=False, 
            pin_memory=(args.device == "cuda"), num_workers=args.workers
        )
        model = SpanExtractionModel(model_name=args.model_name, tokenizer=tokenizer)

        span_train_fn(
            model=model, 
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader, 
            epochs=args.epochs, 
            output_dir=args.output_dir,
            report_to=args.report_to,
            lr=args.lr, 
            logging_step=args.logging_step, 
            precision=args.precision, 
            device=args.device, 
            warmup_ratio=args.warmup_ratio, 
            grad_clip=args.grad_clip
        )
    else:
        train_dataset = WSDDataset(
            dataframe=DataBuilder(args.train_data_dir, args.pos, args.seed).to_pandas(), 
            tokenizer=tokenizer, 
            n_sense=args.num_sense, 
            max_seq_length=args.max_seq_len
        )
        val_dataset = WSDDataset(
            dataframe=DataBuilder(args.val_data_dir, args.pos, args.seed).to_pandas(), 
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
        model = WSDModel(model_name=args.model_name, tokenizer=tokenizer, n_sense=args.num_sense)

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
    if args.report_to == "wandb":
        wandb.finish()

    if args.do_predict:
        def build_predict_args(args: argparse.Namespace, data_dir: str) -> List[str]:
            predict_args = [
                "python", "predict.py",
                "--data_dir", data_dir,
                "--model_name", args.model_name,
                "--output_dir", args.output_dir,
                "--weight_dir", args.output_dir,
                "--pos", args.pos,
                "--num_sense", str(args.num_sense),
                "--max_length", str(args.max_seq_len),
                "--batch_size", str(args.batch_size),
                "--architecture", args.architecture
            ]
            if args.seed is not None:
                predict_args += ["--seed", str(args.seed)]
            return predict_args
        subprocess.run(build_predict_args(args, "data/Evaluation_Datasets/semeval2007"))
        subprocess.run(build_predict_args(args, "data/Evaluation_Datasets/semeval2013"))
        subprocess.run(build_predict_args(args, "data/Evaluation_Datasets/semeval2015"))
        subprocess.run(build_predict_args(args, "data/Evaluation_Datasets/senseval2"))
        subprocess.run(build_predict_args(args, "data/Evaluation_Datasets/senseval3"))

        # Merge all prediction results into a single file
        json_files = glob.glob(os.path.join(args.output_dir, "*.json"))
        dataset_names = [os.path.splitext(os.path.basename(f))[0] for f in json_files]
        output_dir = args.output_dir
        all_metrics = {}

        for dataset in dataset_names:
            json_path = os.path.join(output_dir, f"{dataset}.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    all_metrics[dataset] = json.load(f)
            else:
                raise FileNotFoundError(f"Metrics file for {dataset} not found at {json_path}")
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=4)
        # Plot the metrics
        plot_line(metrics_path, os.path.join(output_dir, "metrics.png"))


parser = argparse.ArgumentParser(description="Word Sense Disambiguation")
parser.add_argument("-c", type=str, required=True, help="Config file path")
args = parser.parse_args()
config_file = args.c 
config = yaml.safe_load(open(config_file, "r", encoding="utf-8"))
args = argparse.Namespace(**config)


if __name__ == "__main__":
    main(args=args)