import json

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import Transformer
from tokenizer import Tokenizer
from config import config, get_model_path, get_vocabs_path
from .dataset import TextDataset


class Trainer:

    def __init__(self, model: Transformer, tokenizer: Tokenizer) -> None:

        self.model = model
        self.tokenizer = tokenizer

        # Initialize dataset
        dataset = TextDataset(tokenizer)

        # Initialize dataloader
        self.dataloader = DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=True
        )

        # Initialize optimizer, cross-entropy loss
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"])
        self.criterion = nn.CrossEntropyLoss(ignore_index=config["pad_i"])

    def train(self):

        # Tensorboard
        writer = SummaryWriter(log_dir=f"runs/{config['experiment_name']}")

        epoch_n = 0
        step = 0

        for epoch in range(config["num_epochs"]):

            # Logger
            loop = tqdm(self.dataloader, desc=f"Epoch {epoch_n}")

            # ids: (batch, seq_len)
            for input_ids, target_ids in loop:

                input_ids = input_ids.to(config["device"])
                target_ids = target_ids.to(config["device"])

                # Clear gradient each round
                self.optimizer.zero_grad()

                logits = self.model(input_ids)  # (batch, seq_len, vocab_size)

                # Loss expects particular shape
                loss: torch.Tensor = self.criterion(
                    # (batch*seq_len, vocab_size)
                    logits.view(-1, config["vocab_size"]),
                    target_ids.view(-1),  # (batch*seq_len)
                )

                # Compute gradients (backprop)
                loss.backward()

                # Clip gradient norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)

                # Nudge parameters based on grad_fn
                self.optimizer.step()

                # Track loss in tensorboard
                writer.add_scalar("Loss", loss.item(), step)
                step += 1

                # Track loss in terminal log
                loop.set_postfix(loss=f"{loss.item():.4f}")

        epoch_n += 1

        # Save model state at end of each epoch
        torch.save(
            {
                "epoch": epoch,  # type: ignore
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": step,
            },
            get_model_path(),
        )

        # Save vocab state
        with open(get_vocabs_path(), "w") as file:
            json.dump(self.tokenizer.vocabs, file)
