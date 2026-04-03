import json

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

# from torch.utils.tensorboard import SummaryWriter

from model import Transformer
from tokenizer import Tokenizer
from config import config
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

        # # Tensorboard
        # writer = SummaryWriter("helo!")

        epoch_n = 0
        step = 0

        for epoch in range(config["num_epochs"]):
            # ids: (batch, seq_len)
            for input_ids, target_ids in self.dataloader:

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

                # Nudge parameters based on grad_fn
                self.optimizer.step()

                # writer.add_scalar("Loss", loss.item())
                # writer.flush()
                step += 1

                print(f"Loss: {loss.item()}, step: {step}, epoch: {epoch_n}")

                torch.save(
                    {
                        "epoch": epoch,  # type: ignore
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "step": step,
                    },
                    f"{config["model_basename"]}.pt",
                )

                # Every 10 steps print current output
                if step % 5 == 0:
                    predicted_ids = torch.argmax(logits, dim=2)  # (batch, seq_len)
                    flat = predicted_ids.view(-1)  # (batch*seq_len)

                    tokens = self.tokenizer.decode(flat.tolist())

                    print(" ".join(tokens))

        epoch_n += 1

        # Save model state at end of each epoch
        torch.save(
            {
                "epoch": epoch,  # type: ignore
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": step,
            },
            f"{config["model_basename"]}.pt",
        )

        # Save vocab state
        with open(config["vocabs_path"], "w") as file:
            json.dump(self.tokenizer.vocabs, file)
