import torch
import transformers as T
from typing import Any, Union, NamedTuple


class ModelOutput(NamedTuple):
    loss: Union[torch.Tensor, None] = None
    logits: Union[torch.Tensor, None] = None


class ContextEncoder(torch.nn.Module):
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super(ContextEncoder, self).__init__()
        self.tokenizer = kwargs.get("tokenizer", None)
        self.bert = T.AutoModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        hidden_repr = outputs.last_hidden_state
        return hidden_repr.mean(dim=1)


class GlossEncoder(torch.nn.Module):
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super(GlossEncoder, self).__init__()
        self.tokenizer = kwargs.get("tokenizer", None)
        self.bert = T.AutoModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        glosses_rep = []
        for gloss, mask in zip(input_ids, attn_mask):
            outputs = self.bert(input_ids=gloss, attention_mask=mask)
            glosses_rep.append(outputs.last_hidden_state.mean(dim=1))
        all_glosses = torch.stack(glosses_rep, dim=0)
        return all_glosses # batch, n_sense, hidden


class WordSenseDisambiguationModel(torch.nn.Module):
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super(WordSenseDisambiguationModel, self).__init__()
        self.tokenizer = kwargs.get("tokenizer", None)
        self.n_sense = kwargs.get("n_sense", 5)
        self.model_name = model_name
        self.context_encoder = ContextEncoder(model_name, **kwargs)
        self.gloss_encoder = GlossEncoder(model_name, **kwargs)
        hidden_size = self.context_encoder.bert.config.hidden_size
        self.embedding = torch.nn.Embedding(len(self.tokenizer), hidden_size)
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, sentence_inputs: torch.Tensor, sentence_mask: torch.Tensor,
            gloss_inputs: torch.Tensor, gloss_mask: torch.Tensor, target_word: torch.Tensor,
            labels: Union[torch.Tensor, None] = None) -> ModelOutput:
        word_embed = self.embedding(target_word).mean(dim=1)
        sent_rep = self.context_encoder(sentence_inputs, sentence_mask)
        sent_rep = sent_rep + word_embed # Enhanced representation
        gloss_rep = self.gloss_encoder(gloss_inputs, gloss_mask)
        sent_rep_exp = sent_rep.unsqueeze(1).expand(-1, gloss_rep.size(1), -1)
        logits = (sent_rep_exp * gloss_rep).sum(dim=-1)
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return ModelOutput(loss=loss, logits=logits)
        return ModelOutput(logits=logits)




if __name__ == "__main__":
    from utils import get_tokenizer
    tokenizer = get_tokenizer("bert-base-uncased")
    batch_size = 2
    seq_len = 50
    n_sense = 5
    gloss_len = 50

    sentence_inputs = torch.randint(0, len(tokenizer), (batch_size, seq_len))
    sentence_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    gloss_inputs = torch.randint(0, len(tokenizer), (batch_size, n_sense, gloss_len))
    gloss_mask = torch.ones((batch_size, n_sense, gloss_len), dtype=torch.long)
    target_word = torch.randint(0, len(tokenizer), (batch_size, 1))
    labels = torch.randint(0, n_sense, (batch_size,))

    model = WordSenseDisambiguationModel("bert-base-uncased", tokenizer=tokenizer)
    output = model(sentence_inputs, sentence_mask, gloss_inputs, gloss_mask, labels)
    print(output.shape)



