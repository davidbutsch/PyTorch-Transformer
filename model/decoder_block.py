import torch
import torch.nn as nn
from .residual_connection import ResidualConnection
from .attention import MaskedMultiHeadAttention
from .feedforward import Feedforward


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, h: int, max_seq_len: int) -> None:
        super().__init__()
        self.residual_connection = ResidualConnection(
            sub_layers=[
                MaskedMultiHeadAttention(d_model, h, max_seq_len),
                Feedforward(d_model, d_ff),
            ],
            d_model=d_model,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.residual_connection(embeddings)
