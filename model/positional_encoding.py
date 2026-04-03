import torch
from torch import nn
from config import config
import math


class PositionalEncoding(nn.Module):
    def __init__(self, N: int = 10000):
        super().__init__()

        # Generate max_seq_len sinusodial positional vectors in R^d_model
        PE = torch.empty(
            1,  # batch dimension set to 1 for broadcast
            config["max_seq_len"],
            config["d_model"],
            requires_grad=False,  # We don't change these values -> constant buffer does not compute gradient
        )

        for p in range(config["max_seq_len"]):
            # Each i refers to a "pair" dimension -> PE_i: [0, 0, 2, 2, 4, 4]
            for i in range(0, config["d_model"], 2):
                n_i = N ** ((2 * i) / config["d_model"])  # n_i = N^2i/d_model
                ratio = p / n_i

                # Per the formula @ sect 3.5 [https://arxiv.org/pdf/1706.03762]
                PE[0, p, i] = math.sin(ratio)
                PE[0, p, i + 1] = math.cos(ratio)

        self.register_buffer("PE", PE)  # (batch, max_seq_len, d_model)

    # Input embeddings: (batch, seq_len, d_model)
    def forward(self, embeddings: torch.Tensor):
        seq_len = embeddings.shape[1]
        PE = self.get_buffer("PE")
        sliced_PE = PE[
            :,  # batch dimension
            :seq_len,  # Slice sinusodial max_seq_len dimension -> only include positional vectors in seq_len
            :,  # d_model dimension
        ]

        return embeddings + sliced_PE  # (batch, seq_len, d_model)
