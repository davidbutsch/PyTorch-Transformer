import math
import json

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LambdaLR

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
        self.optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=config["max_lr"]
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=config["pad_i"])

    def _get_lr_scheduler(self, warmup_steps: int, total_steps: int):
        # lr_lambda returns factor we apply to the base learning rate
        # actual_lr = base_lr * lr_lambda(current_step)
        def lr_lambda(current_step):
            # If in warmup phase...
            if current_step < warmup_steps:
                # Linear warmup from 0 -> 1
                return float(current_step) / float(max(1, warmup_steps))

            # Cosine decay (only the decreasing part of the cosine curve)

            # Map current step to: [0, 1]: {warmup_steps -> 0, total_steps -> 1}
            # (x - min) / (max - min)
            t: float = (current_step - warmup_steps) / (total_steps - warmup_steps)

            min_lr = config["min_lr"]

            # Generate cosine curve where cos(0) = 1, cos(pi) = min_lr {0: 1, pi: min_lr}
            # https://www.desmos.com/calculator/c8uvidvptr
            # Below is standard cosine decay formula
            return min_lr + (1 - min_lr) * 0.5 * (1 + math.cos(t * math.pi))

        return LambdaLR(self.optimizer, lr_lambda)

    def train(self):

        # Tensorboard
        writer = SummaryWriter(log_dir=f"runs/{config['experiment_name']}")

        # Use learning rate scheduler
        total_steps = len(self.dataloader) * config["num_epochs"]
        warmup_steps = int(config["warmup_ratio"] * total_steps)
        scheduler = self._get_lr_scheduler(warmup_steps, total_steps)

        step = 0

        for epoch in range(config["num_epochs"]):

            # Logger
            loop = tqdm(self.dataloader, desc=f"Epoch {epoch}")

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

                scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Track metrics in tensorboard
                writer.add_scalar("Loss", loss.item(), step)
                writer.add_scalar("Learning_Rate", current_lr, step)

                # Track loss in terminal log
                loop.set_postfix(loss=f"{loss.item():.4f}")
                loop.set_description(f"lr={current_lr:.2e}")

                step += 1

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
