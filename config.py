import torch
from pathlib import Path

# Model architecture constants — shared by model files, tokenizer, and inference.
# Training hyperparameters (lr, batch size, warmup, etc.) live in train.py.
config = {
    "vocab_size": 2**14,
    "special_tokens": {
        "<|endoftext|>": 2**14 - 1,
    },
    "max_seq_len": 512,
    "d_model": 256,
    "d_ff": 1024,
    "h": 8,
    "N": 12,
    "tokenizer_pattern": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
    "tokenizer_prefix": "2pow14",
    "dataset_prefix": "openwebtext",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def _get_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_dataset_path() -> Path:
    return _get_dir(Path(__file__).parent / "data") / f"{config['dataset_prefix']}.bin"


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


def get_checkpoint_path() -> Path:
    """nanoGPT-style checkpoint written by train.py."""
    return _get_dir(Path(__file__).parent / "out") / "ckpt.pt"
