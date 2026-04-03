import torch
import torch.nn as nn
from config import config


class Embedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # T: R^1 -> R^d_model
        # token_id -> embedding vector of length d_model
        self.embedding_layer = nn.Embedding(
            num_embeddings=config["vocab_size"],
            embedding_dim=config["d_model"],
            padding_idx=config["pad_i"],
        )

    # Input token_ids: (batch, seq_len, token_id) where token_id is size 1
    def forward(self, token_ids: torch.IntTensor):
        return self.embedding_layer(token_ids)  # (batch, seq_len, d_model)
