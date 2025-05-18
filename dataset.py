import torch
import pandas as pd
from transformers import AutoTokenizer
from typing import Optional, Dict, Any, Tuple, List, Union, Callable

class WSDDataset(torch.utils.data.Dataset):
    """
    Dataset for Word Sense Disambiguation (WSD) task.
    """
    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer, n_sense: int = 5,
            max_seq_length: int = 128) -> None:
        self.df = dataframe
        self.tokenizer = tokenizer
        self.n_sense = n_sense
        self.max_seq_length = max_seq_length

    def generate_features(self, row: pd.Series) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        sentence = row['sentence']
        start = row['target_index_start']
        end = row['target_index_end']
        correct_sense = row['correct_sense']
        target_word = row["target_lemma"]

        base_senses = row.get('sense_list', [])[: max(0, self.n_sense - 1)]
        if correct_sense not in base_senses:
            base_senses.append(correct_sense)

        senses = base_senses[-self.n_sense:]
        assert len(senses) <= self.n_sense, f"Too many senses: {len(senses)}"
        senses = senses + [self.tokenizer.pad_token] * (self.n_sense - len(senses))

        # Reconstruct target span and insert special tokens
        tokens = sentence.split()
        target_span = " ".join(tokens[start:end])
        special_token = self.add_special_tokens(target_span)
        input_text = sentence.replace(target_span, special_token, 1)

        # Convert to token ids
        encodded_sentence = self.tokenizer(
            input_text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_seq_length
            )
        sentence_inputs = encodded_sentence["input_ids"].squeeze(0)
        sentence_mask = encodded_sentence["attention_mask"].squeeze(0)

        glosses_inputs, glosses_mask = [], []
        for sense in senses:
            encodded_sense = self.tokenizer(
                sense, return_tensors="pt", padding="max_length",
                truncation=True, max_length=self.max_seq_length
                )
            glosses_inputs.append(encodded_sense["input_ids"].squeeze(0))
            glosses_mask.append(encodded_sense["attention_mask"].squeeze(0))

        assert len(glosses_inputs) == len(glosses_mask) == self.n_sense
        target_word = self.tokenizer.encode(
            target_word, add_special_tokens=False, max_length=5, truncation=True,
            padding="max_length", return_tensors="pt"
            ).squeeze(0)
        labels = senses.index(correct_sense)
        return {
            "sentence_inputs": sentence_inputs,
            "sentence_mask": sentence_mask,
            "gloss_inputs": torch.stack(glosses_inputs),
            "gloss_mask": torch.stack(glosses_mask),
            "target_word": target_word,
            "labels": torch.tensor(labels),
        }

    def add_special_tokens(self, word: str) -> str:
        return f"<classify>{word}</classify>"

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        row = self.df.iloc[index]
        return self.generate_features(row)

    def __len__(self) -> int:
        return len(self.df)


__all__ = ["WSDDataset"]