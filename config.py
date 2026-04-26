import torch
from pathlib import Path

config = {
    "vocab_size": 128,  # Number of learned token embeddings
    "max_seq_len": 512,  # Max input token sequence length
    "d_model": 256,  # Embedding dimension
    "d_ff": 1024,  # Feedforward network hidden dimension (4*d_model)
    "h": 8,  # Number of attention heads
    "N": 6,  # Number of decoder blocks
    "pad_i": 0,  # Padding token index
    "num_epochs": 32,  # Number of iterations through complete data set
    "batch_size": 64,  # Number of sentences per training batch
    "max_lr": 1e-3,  # Maximum learning rate (after warmup)
    "min_lr": 1e-5,  # Minimum learning rate (end of cosine decay)
    "warmup_ratio": 0.05,  # Define what % of steps are warmup steps
    "ds_path": Path(__file__).parent / "training" / "tiny_shakespeare.txt",
    "experiment_name": "logger_test",
    "saves_path": Path(__file__).parent / "saves",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def _get_saves_dir() -> Path:
    saves_dir = Path(config["saves_path"])
    saves_dir.mkdir(parents=True, exist_ok=True)
    return saves_dir


def get_vocabs_path() -> Path:
    return _get_saves_dir() / f"vocabs_{config['experiment_name']}.json"


def get_model_path() -> Path:
    return _get_saves_dir() / f"model_{config['experiment_name']}.pt"
