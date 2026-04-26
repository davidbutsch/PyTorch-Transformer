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
    "num_epochs": 128,  # Number of iterations through complete data set
    "batch_size": 64,  # Number of sentences per training batch
    "max_lr": 1e-3,  # Maximum learning rate (after warmup)
    "min_lr_ratio": 0.1,  # Minimum learning rate as fraction of max_lr (e.g., 0.1 = decay to 10% of peak)
    "warmup_ratio": 0.05,  # Define what % of steps are warmup steps
    "experiment_name": "book_corpus",
    "dataset": "rojagtap/bookcorpus",
    "dataset_name": "bookcorpus",
    "saves_path": Path(__file__).parent / "saves",
    "datasets_path": Path(__file__).parent / "datasets",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def _get_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_preprocessed_ds_path() -> Path:
    return _get_dir(config["datasets_path"]) / f"{config['dataset_name']}_tokens.bin"


def get_vocabs_path() -> Path:
    return _get_dir(config["datasets_path"]) / f"{config['dataset_name']}_vocabs.json"


def get_model_path() -> Path:
    return _get_dir(config["saves_path"]) / f"model_{config['experiment_name']}.pt"
