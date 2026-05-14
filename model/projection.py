import torch.nn as nn


class Projection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        # T: R^d_model -> R^vocab_size
        self.projection_layer = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, c_embeddings):
        return self.projection_layer(c_embeddings)  # (batch, seq_len, vocab_size)
