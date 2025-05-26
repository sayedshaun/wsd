import os
import torch
import argparse
from dataset import WSDDataset
from data_builder import DataBuilder
from model import WordSenseDisambiguationModel
from utils import get_tokenizer
from torch.utils.data import DataLoader
from train_utils import evaluation_fn


def main(args: argparse.Namespace):
    tokenizer = get_tokenizer(args.model_name)
    model = WordSenseDisambiguationModel(args.model_name, tokenizer=tokenizer).to(args.device)
    weight_list = os.listdir(args.weight_dir)
    path = sorted(weight_list, key=lambda x: x.split("-")[-1])[-1]
    print("Loading model from: ", path)
    state_dict = torch.load(os.path.join(args.weight_dir, path), map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    ds = WSDDataset(
        dataframe=DataBuilder(args.data_dir, args.pos_tag).to_pandas(), 
        tokenizer=tokenizer, 
        n_sense=args.num_sense, 
        max_seq_length=args.max_length,
        shuffle_senses=False
    )
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    loss, f1, precision, recall, acc = evaluation_fn(model, dataloader, args.device)
    print("Loss: ", loss)
    print("F1: ", f1)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Accuracy: ", acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--weight_dir", type=str, required=True, help="Path to model weight directory")
    parser.add_argument("--pos_tag", type=str, default="ALL", help="POS tag to use")
    parser.add_argument("--num_sense", type=int, default=5, help="Number of sense")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of input")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    main(args)