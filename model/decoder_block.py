import torch
import torch.nn as nn
from .residual_connection import ResidualConnection
from .attention import MaskedAttention
from .feedforward import Feedforward


class DecoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Each decoder block applies a residually connected attention -> ffn
        self.residual_connection = ResidualConnection(
            [
                MaskedAttention(),
                Feedforward(),
            ]
        )

    # Input embeddings: (batch, seq_len, d_model)
    def forward(self, embeddings: torch.Tensor):
        return self.residual_connection(embeddings)  # (batch, seq_len, d_model)
