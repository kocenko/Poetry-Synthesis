import torch
import torch.nn as nn


class OnlyDecoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.vocab_size = config["vocab_size"]
        self.embedding_table = nn.Embedding(self.vocab_size, self.vocab_size)

    def forward(self, token_idx: int, targets: torch.tensor) -> torch.tensor:
        logits = self.embedding_table(token_idx)
        return logits