import os

import torch

from config import config, get_model_path
from model import Transformer
from tokenizer import Tokenizer
from training import Trainer


def train():

    print(f"Training on {config['device']}")

    # Load model state from disk
    if os.path.exists(get_model_path()):
        print("Loading saved model...")
        state = torch.load(get_model_path())
    else:
        state = None

    model = Transformer(state).to(config["device"])
    tokenizer = Tokenizer()

    trainer = Trainer(model, tokenizer, state)

    trainer.train()


if __name__ == "__main__":
    train()
