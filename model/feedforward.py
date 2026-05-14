import torch
import torch.nn as nn


class Feedforward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_ff),   # T: R^d_model -> R^d_ff
            nn.Linear(d_ff, d_model),   # T: R^d_ff   -> R^d_model
        ])

    def forward(self, c_embeddings: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            c_embeddings = layer(c_embeddings)
            if i < len(self.layers) - 1:
                # Skip activation + dropout on the last layer; the residual
                # connection normalizes the output instead.
                c_embeddings = self.dropout(self.act(c_embeddings))
        return c_embeddings
