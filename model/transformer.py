import os

import torch
import torch.nn as nn

from .embedding import Embedding
from .positional_encoding import PositionalEncoding
from .decoder_block import DecoderBlock
from .projection import Projection

from config import config


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        # Instantiate layers
        self.embedding = Embedding()
        self.positional_encoding = PositionalEncoding()
        self.decoders = nn.ModuleList([DecoderBlock() for _ in range(config["N"])])
        self.projection = Projection()

        # Load model state from disk
        if os.path.exists(f"{config["model_basename"]}.pt"):
            print("Loading saved model...")
            state = torch.load(f"{config["model_basename"]}.pt")
            self.load_state_dict(state["model_state_dict"])

    # Input token ids: (batch, seq_len)
    def forward(self, input_ids: torch.IntTensor):

        # Embeddings
        embeddings: torch.Tensor = self.embedding(
            input_ids
        )  # (batch, seq_len, d_model)

        # Positional encodings
        embeddings = self.positional_encoding(embeddings)  # (batch, seq_len, d_model)

        # Decoders
        for decoder in self.decoders:
            embeddings = decoder(embeddings)  # (batch, seq_len, d_model)

        # Project to logits
        logits: torch.Tensor = self.projection(
            embeddings
        )  # (batch, seq_len, vocab_size)

        return logits
