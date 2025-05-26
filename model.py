import torch
import transformers as T
from typing import Any, Union, NamedTuple


class ModelOutput(NamedTuple):
    loss: Union[torch.Tensor, None] = None
    logits: Union[torch.Tensor, None] = None


class WordSenseDisambiguationModel(torch.nn.Module):
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super(WordSenseDisambiguationModel, self).__init__()
        self.tokenizer = kwargs.get("tokenizer", None)
        self.n_sense = kwargs.get("n_sense", 5)
        self.model_name = model_name
        self.encoder = T.AutoModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(
            len(self.tokenizer), mean_resizing=False
        )

    def forward(
        self: "WordSenseDisambiguationModel",
        sentence_inputs: torch.Tensor,   # [B, T]
        sentence_mask:  torch.Tensor,    # [B, T]
        gloss_inputs:   torch.Tensor,    # [B, S, L]
        gloss_mask:     torch.Tensor,    # [B, S, L]
        target_word:    torch.Tensor,    # [B, W]
        labels:         torch.Tensor=None
        ) -> ModelOutput:
        
        context_encoder = self.encoder(sentence_inputs, sentence_mask)
        context_rep = context_encoder.last_hidden_state[:, 0, :]
        context_rep = context_rep.unsqueeze(1).expand(-1, self.n_sense, -1)
        
        B, S, L = gloss_inputs.size()  # [B, S, L]
        gloss_in  = gloss_inputs.view(B * S, L)             # [B*S, L]
        gloss_msk = gloss_mask.view(B * S, L)               # [B*S, L]
        gloss_enc = self.encoder(gloss_in, gloss_msk)       
        gloss_cls = gloss_enc.last_hidden_state[:, 0, :]    # [B*S, H]
        gloss_rep = gloss_cls.view(B, S, -1)                # [B, S, H]

        logits = torch.einsum("bsh,bsh->bs", context_rep, gloss_rep)     
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return ModelOutput(loss=loss, logits=logits)
        else:
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



