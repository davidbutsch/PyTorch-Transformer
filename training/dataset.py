import torch
from torch.utils.data import Dataset
from tokenizer import Tokenizer
from config import config


class TextDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__()

        self.tokenized_dataset = []

        # Tokenize text file lines
        with open(config["ds_path"], "r") as file:

            for line in file:

                stripped_line = line.strip()

                # Tokenize
                normalized_line = tokenizer.normalize(stripped_line)  # Normalize line
                tokens = tokenizer.tokenize(normalized_line)  # Tokenize normalized line
                tokenizer.build_vocab(tokens)  # Add to vocabulary with line tokens
                token_ids = tokenizer.encode(tokens)  # Get token ids from tokens

                # Keep tokenized_line
                self.tokenized_dataset.extend(token_ids)

    def __len__(self):
        return len(self.tokenized_dataset) // config["max_seq_len"]

    # Returns (input_ids, target_ids)
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:

        # Read in token ids in range [index * max_seq_len, (index + 1) * max_seq_len + 1]

        # Read in token ids
        token_ids: list[int] = self.tokenized_dataset[
            index * config["max_seq_len"] : (index + 1) * config["max_seq_len"] + 1
        ]

        # Add {pad_len} padding tokens to reach max_seq_len (if necessary)
        pad_len = config["max_seq_len"] + 1 - len(token_ids)
        token_ids = token_ids + [config["pad_i"]] * pad_len

        # Build target, input tensors
        input_ids = torch.tensor(token_ids[:-1])  # e.g. [A, B]
        target_ids = torch.tensor(token_ids[1:])  # e.g. [B, C]

        # If    max_seq_len = 2,
        #       token_ids = [A, B, C]   -> 3 tokens (?!)

        # then  input_ids = [A, B],     -> 2 tokens
        #       target_ids = [B, C]     -> 2 tokens :)

        return input_ids, target_ids
