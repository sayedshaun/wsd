import torch
import transformers as T
import torch.nn.functional as F
from typing import Any, Union, NamedTuple


class ModelOutput(NamedTuple):
    """A named tuple to hold model outputs."""
    loss: Union[torch.Tensor, None] = None
    logits: Union[torch.Tensor, None] = None
    hidden_states: Union[torch.Tensor, None] = None
    attentions: Union[torch.Tensor, None] = None


class QAModelOutput(NamedTuple):
    """A named tuple to hold outputs for QA tasks."""
    loss: Union[torch.Tensor, None] = None
    start_logits: Union[torch.Tensor, None] = None
    end_logits: Union[torch.Tensor, None] = None
    hidden_states: Union[torch.Tensor, None] = None
    attentions: Union[torch.Tensor, None] = None


class WSDModel(torch.nn.Module):
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super(WSDModel, self).__init__()
        self.tokenizer = kwargs.get("tokenizer", None)
        self.n_sense = kwargs.get("n_sense", 5)
        self.model_name = model_name
        self.encoder = T.AutoModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(
            len(self.tokenizer), mean_resizing=False
        )

    def forward(
        self: "WSDModel",
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


class SpanExtractionModel(torch.nn.Module):
    """Model for span extraction tasks."""
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super(SpanExtractionModel, self).__init__()
        self.tokenizer = kwargs.get("tokenizer", None)
        self.model_name = model_name
        self.encoder = T.AutoModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(
            len(self.tokenizer), mean_resizing=False
        )
        hidden_size = self.encoder.config.hidden_size
        self.start_classifier = torch.nn.Linear(hidden_size, 1)
        self.end_classifier = torch.nn.Linear(hidden_size, 1)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(
        self: "SpanExtractionModel",
        input_ids: torch.Tensor,  # [batch_size, seq_length]
        attention_mask:   torch.Tensor,  # [batch_size, seq_length]
        start_positions: torch.Tensor=None,  # [batch_size]
        end_positions:   torch.Tensor=None   # [batch_size]
        ) -> ModelOutput:
        encoder_output = self.encoder(input_ids, attention_mask)
        sequence_output = encoder_output.last_hidden_state  # [batch_size, seq_length, hidden_size]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_classifier(sequence_output).squeeze(-1)  # [batch_size, seq_length]
        end_logits = self.end_classifier(sequence_output).squeeze(-1)      # [batch_size, seq_length]
        if start_positions is not None and end_positions is not None:
            batch_size, seq_length = start_logits.size()
            start_positions = start_positions.clamp(0, seq_length)
            end_positions = end_positions.clamp(0, seq_length)
            start_loss = F.cross_entropy(
                start_logits, start_positions, reduction='mean', ignore_index=seq_length)
            end_loss = F.cross_entropy(
                end_logits, end_positions, reduction='mean', ignore_index=seq_length)
            loss = (start_loss + end_loss) / 2
            return QAModelOutput(
                loss=loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=encoder_output.hidden_states,
                attentions=encoder_output.attentions
            )
        else:
            return ModelOutput(
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=encoder_output.hidden_states,
                attentions=encoder_output.attentions
            )
        


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

    model = WSDModel("bert-base-uncased", tokenizer=tokenizer)
    output = model(sentence_inputs, sentence_mask, gloss_inputs, gloss_mask, labels)
    print(output.shape)



