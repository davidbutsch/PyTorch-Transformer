import torch
import torch.nn as nn

from config import config


class Projection(nn.Module):
    def __init__(self):
        super().__init__()

        # T: R^d_model -> R^vocab_size
        # Transforms decoder output to vocab logits
        self.projection_layer = nn.Linear(
            in_features=config["d_model"], out_features=config["vocab_size"]
        )

    # Input c_embeddings: (batch, seq_len, d_model)
    # Output logits: (batch, seq_len, vocab_size)
    def forward(self, c_embeddings: torch.Tensor):
        return self.projection_layer(c_embeddings)
