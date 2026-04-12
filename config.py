import torch
from pathlib import Path

config = {
    "vocab_size": 100,  # Number of learned token embeddings
    "max_seq_len": 512,  # Max input token sequence length
    "d_model": 256,  # Embedding dimension
    "d_ff": 1024,  # Feedforward network hidden dimension (4*d_model)
    "h": 8,  # Number of attention heads
    "N": 6,  # Number of decoder blocks
    "pad_i": 0,  # Padding token index
    "num_epochs": 6,  # Number of iterations through complete data set
    "batch_size": 16,  # Number of sentences per training batch
    "lr": 1e-3,  # Learning rate ~ adjustment step size
    "ds_path": Path(__file__).parent / "training" / "tiny_shakespeare.txt",
    "vocabs_path": Path(__file__).parent / "saves" / "vocabs_char_level.json",
    "model_basename": Path(__file__).parent / "saves" / "model_char_level",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
