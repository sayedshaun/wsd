import os
import torch
import argparse
from dataset import WSDDataset, SpanDataset
from data_builder import DataBuilder
from model import WSDModel, SpanExtractionModel
from utils import get_tokenizer, load_model_weights
from torch.utils.data import DataLoader
from train_utils import evaluation_fn, span_evaluation_fn




def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(args.model_name)
    if args.architecture == "cosine_similarity":
        model = WSDModel(args.model_name, tokenizer=tokenizer).to(device)
        model = load_model_weights(model, args.weight_dir, device)
        ds = WSDDataset(
            dataframe=DataBuilder(args.data_dir, args.pos, args.seed).to_pandas(), 
            tokenizer=tokenizer, 
            n_sense=args.num_sense, 
            max_seq_length=args.max_length,
            shuffle_senses=False
        )
        dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        loss, f1, precision, recall, acc = evaluation_fn(model, dataloader, device)
        print("=" * 50)
        print("Loss: ", loss)
        print("F1: ", f1)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("Accuracy: ", acc)
        print("=" * 50)
    else:
        model = SpanExtractionModel(args.model_name, tokenizer=tokenizer).to(device)
        model = load_model_weights(model, args.weight_dir, args.device)
        ds = SpanDataset(
            dataframe=DataBuilder(args.data_dir, args.pos, args.seed).to_pandas(), 
            tokenizer=tokenizer, 
            max_seq_length=args.max_length
        )
        dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        loss, start_accu, end_accu, joint_accu = span_evaluation_fn(model, dataloader, device)
        print("=" * 50)
        print("Loss: ", loss)
        print("Start Accuracy: ", start_accu)
        print("End Accuracy: ", end_accu)
        print("Joint Accuracy: ", joint_accu)
        print("=" * 50)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--weight_dir", type=str, required=True, help="Path to model weight directory")
    parser.add_argument("--pos", type=str, default="ALL", help="POS tag to use")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")
    parser.add_argument("--num_sense", type=int, default=5, help="Number of sense")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of input")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--architecture", required=True, type=str, help="Model architecture to use")
    args = parser.parse_args()
    main(args)