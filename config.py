import torch
from pathlib import Path

config = {
    "vocab_size": 2**14,  # Number of learned token embeddings
    "special_tokens": {
        "<|endoftext|>": 2**14 - 1,
    },
    "max_seq_len": 512,  # Max input token sequence length
    "d_model": 256,  # Embedding dimension
    "d_ff": 1024,  # Feedforward network hidden dimension (4*d_model)
    "h": 8,  # Number of attention heads
    "N": 6,  # Number of decoder blocks
    "total_steps": 40,  # Number of training steps to perform
    "batch_size": 2,  # Number of examples per training batch
    "max_lr": 1e-3,  # Maximum learning rate (after warmup)
    "min_lr_ratio": 0.1,  # Minimum learning rate as fraction of max_lr (e.g., 0.1 = decay to 10% of peak)
    "warmup_ratio": 0.05,  # Define what % of steps are warmup steps
    "dataset": "Skylion007/openwebtext",
    "tokenizer_prefix": "2pow14",
    "model_prefix": "openwebtext_model_1",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def _get_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_merges_path() -> Path:
    return (
        _get_dir(Path(__file__).parent / "tokenizer" / "saves")
        / f"{config['tokenizer_prefix']}.merges"
    )


def get_vocabs_path() -> Path:
    return (
        _get_dir(Path(__file__).parent / "tokenizer" / "saves")
        / f"{config['tokenizer_prefix']}.vocabs"
    )


def get_model_path() -> Path:
    return (
        _get_dir(Path(__file__).parent / "model" / "saves")
        / f"{config['model_prefix']}.pt"
    )
