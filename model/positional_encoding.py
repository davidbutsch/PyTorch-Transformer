import math
import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    PE: Tensor

    def __init__(self, max_seq_len: int, d_model: int, N: int = 10000) -> None:
        super().__init__()

        # Pre-compute sinusoidal positional vectors for all positions.
        # Formula from Attention Is All You Need, sect. 3.5:
        # https://arxiv.org/pdf/1706.03762
        PE = torch.empty(1, max_seq_len, d_model, requires_grad=False)
        for p in range(max_seq_len):
            for i in range(0, d_model, 2):
                n_i = N ** ((2 * i) / d_model)
                ratio = p / n_i
                PE[0, p, i] = math.sin(ratio)
                PE[0, p, i + 1] = math.cos(ratio)

        # Registered as a buffer: moves with the model, no gradient computed.
        self.register_buffer("PE", PE)  # (1, max_seq_len, d_model)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        seq_len = embeddings.shape[1]
        return embeddings + self.PE[:, :seq_len, :]  # (batch, seq_len, d_model)
