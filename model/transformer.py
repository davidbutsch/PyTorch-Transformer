import torch
import torch.nn as nn

from .embedding import Embedding
from .positional_encoding import PositionalEncoding
from .decoder_block import DecoderBlock
from .projection import Projection

from config import config  # only model file that reads config


class Transformer(nn.Module):
    """Decoder-only transformer. All architectural dimensions are explicit
    constructor params with defaults pulled from config.py, so you can
    experiment inline without touching config:

        Transformer()               # uses config defaults
        Transformer(N=4, d_model=128)  # quick smaller experiment
    """

    def __init__(
        self,
        vocab_size:  int = config["vocab_size"],
        d_model:     int = config["d_model"],
        d_ff:        int = config["d_ff"],
        h:           int = config["h"],
        N:           int = config["N"],
        max_seq_len: int = config["max_seq_len"],
    ) -> None:
        super().__init__()
        self.embedding          = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model)
        self.decoders           = nn.ModuleList([DecoderBlock(d_model, d_ff, h, max_seq_len) for _ in range(N)])
        self.projection         = Projection(d_model, vocab_size)

    # Input token ids: (batch, seq_len)
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)             # (batch, seq_len, d_model)
        embeddings = self.positional_encoding(embeddings)  # (batch, seq_len, d_model)
        for decoder in self.decoders:
            embeddings = decoder(embeddings)               # (batch, seq_len, d_model)
        return self.projection(embeddings)                 # (batch, seq_len, vocab_size)
