import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import Transformer
from tokenizer import RegexTokenizer
from config import config, get_model_path
from .dataset import PackedDataset


class ModelTrainer:

    def __init__(
        self, model: Transformer, tokenizer: RegexTokenizer, state=None
    ) -> None:

        self.model = model
        self.tokenizer = tokenizer

        # Initialize dataset
        self.iterable_dataset = PackedDataset(
            tokenizer,
            batch_size=config["batch_size"],
            max_seq_len=config["max_seq_len"],
        )

        # Initialize optimizer, cross-entropy loss
        self.optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=config["max_lr"]
        )
        self.criterion = nn.CrossEntropyLoss()

        if state is not None:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.start_step = state["step"]
        else:
            self.start_step = 0

        # Use learning rate scheduler

        # Compute total steps with SPH, total training hours...
        self.total_steps = config["total_steps"] - self.start_step
        warmup_steps = int(config["warmup_ratio"] * self.total_steps)
        self.scheduler = self._get_lr_scheduler(warmup_steps, self.total_steps)

        # This is hacky but `_get_lr_scheduler` needs to know about `self.start_step` above
        if state is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

    def _get_lr_scheduler(self, warmup_steps: int, total_steps: int):
        assert (
            total_steps > 0
        ), "Total steps must be greater than 0. Check your training hours."

        # lr_lambda returns factor we apply to the base learning rate
        # actual_lr = base_lr * lr_lambda(current_step)
        def lr_lambda(current_step):
            # If in warmup phase...
            if current_step + self.start_step < warmup_steps:
                # Linear warmup from 0 -> 1
                return float(current_step) / float(max(1, warmup_steps))

            # Cosine decay (only the decreasing part of the cosine curve)

            # Map current step to: [0, 1]: {warmup_steps -> 0, total_steps -> 1}
            # (x - min) / (max - min)
            t: float = (current_step - warmup_steps) / (total_steps - warmup_steps)

            min_lr_ratio = config["min_lr_ratio"]

            # Generate cosine curve where cos(0) = 1, cos(pi) = min_lr {0: 1, pi: min_lr}
            # https://www.desmos.com/calculator/c8uvidvptr
            # Below is standard cosine decay formula
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(t * math.pi))

        return LambdaLR(self.optimizer, lr_lambda)

    def _save_checkpoint(self, step: int) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "step": self.start_step + step,
            },
            get_model_path(),
        )

    def train(self):
        # Tensorboard
        writer = SummaryWriter(log_dir=f"runs/{config['model_prefix']}")

        step = 0

        print(f"Training from step {self.start_step}")

        loop = tqdm(
            self.iterable_dataset,
            desc=f"overall_step={self.start_step + step}, lr={self.optimizer.param_groups[0]['lr']:.2e}",
            postfix={"loss": f"--.----"},
            total=self.total_steps,
        )

        try:

            for example in loop:

                # ids: (batch_size, max_seq_len)
                input_ids, target_ids = example

                input_ids = input_ids.to(config["device"])
                target_ids = target_ids.to(config["device"])

                # Clear gradient each round
                self.optimizer.zero_grad()

                logits = self.model(input_ids)  # (batch_size, max_seq_len, vocab_size)

                # Loss expects particular shape
                loss: torch.Tensor = self.criterion(
                    # (batch_size*seq_len, vocab_size)
                    logits.view(-1, config["vocab_size"]),
                    target_ids.reshape(-1),  # (batch_size*seq_len)
                )

                # Compute gradients (backprop)
                loss.backward()

                # Clip gradient norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)

                # Nudge parameters based on grad_fn
                self.optimizer.step()

                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Track metrics in tensorboard
                writer.add_scalar("Loss", loss.item(), step)
                writer.add_scalar("Learning_Rate", current_lr, step)

                step += 1

                # Track loss in terminal log
                loop.set_postfix(loss=f"{loss.item():.4f}")
                loop.set_description(
                    f"overall_step={self.start_step + step}, lr={current_lr:.2e}"
                )

                # Exit after reaching total steps
                if step >= self.total_steps:
                    self._save_checkpoint(step)
                    return

        except KeyboardInterrupt:
            print(f"Training interrupted at step {step}")
            self._save_checkpoint(step)
        finally:
            writer.close()
