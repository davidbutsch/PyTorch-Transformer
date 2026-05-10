import os

import torch
from datasets import load_dataset

from config import config, get_merges_path, get_model_path
from model import Transformer
from tokenizer import RegexTokenizer
from model.training import ModelTrainer


def train():

    print(f"Training on {config['device']}")

    # Load transformer model state from disk
    if os.path.exists(get_model_path()):
        print("Loading saved model...")
        state = torch.load(get_model_path())
    else:
        state = None

    model = Transformer(state).to(config["device"])

    tokenizer = RegexTokenizer()

    # Load tokenizer model state from disk
    if os.path.exists(get_merges_path()):
        print("Loading saved tokenizer...")
        tokenizer.load()
    else:
        # Train tokenizer if save not found
        print("Training tokenizer...")

        ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
        train_text = "\n\n".join(
            ex["text"] for ex in ds if isinstance(ex, dict) and ex["text"].strip()
        )
        train_text = train_text[:5_000_000]

        tokenizer.register_pattern(config["tokenizer_pattern"])
        tokenizer.register_special_tokens(config["special_tokens"])

        tokenizer.train(train_text, config["vocab_size"])

    trainer = ModelTrainer(model, tokenizer, state)
    trainer.train()


if __name__ == "__main__":
    train()
