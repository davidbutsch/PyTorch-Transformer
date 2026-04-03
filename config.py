from pathlib import Path

config = {
    "vocab_size": 32000,  # Number of learned token embeddings
    "max_seq_len": 512,  # Max input token sequence length
    "d_model": 64,  # Embedding dimension
    "d_ff": 256,  # Feedforward network hidden dimension (4*d_model)
    "h": 1,  # Number of attention heads
    "N": 12,  # Number of decoder blocks
    "pad_i": 0,  # Padding token index
    "num_epochs": 6,  # Number of iterations through complete data set
    "batch_size": 16,  # Number of sentences per training batch
    "lr": 1e-4,  # Learning rate ~ adjustment step size
    "ds_path": Path(__file__).parent / "training" / "tiny_shakespeare.txt",
    "vocabs_path": Path(__file__).parent / "vocabs.json",
    "model_basename": Path(__file__).parent / "model",
}
