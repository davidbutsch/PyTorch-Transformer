import torch
import torch.nn as nn
from config import config


class Feedforward(nn.Module):

    def __init__(self, dropout_rate=0.1) -> None:
        super().__init__()

        # Initialize activation fx, dropout, and hidden layers

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        self.layers = nn.ModuleList(
            [
                # T: R^d_model -> R^d_ff
                # Input layer transforms embedding dimension to hidden dimension
                nn.Linear(in_features=config["d_model"], out_features=config["d_ff"]),
                # T: R^d_ff -> R^d_model
                # Output layer transforms hidden dimension to embedding dimension
                nn.Linear(in_features=config["d_ff"], out_features=config["d_model"]),
                # No activation function
            ]
        )

    # Input contextual embeddings: (batch, seq_len, d_model)
    def forward(self, c_embeddings: torch.Tensor):
        for i, layer in enumerate(self.layers):
            c_embeddings = layer(c_embeddings)
            # No activation/dropout applied after last layer -> output is normalized in residual connection
            if i < len(self.layers) - 1:
                # Non-linear activation function prevents linear collapse
                c_embeddings = self.act(c_embeddings)
                # Dropout prevents overfitting
                c_embeddings = self.dropout(c_embeddings)

        return c_embeddings
