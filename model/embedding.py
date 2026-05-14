import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        # T: token_id -> R^d_model
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, token_ids):
        return self.embedding_layer(token_ids)  # (batch, seq_len, d_model)
