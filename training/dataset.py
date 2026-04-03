import torch
from torch.utils.data import Dataset
from tokenizer import Tokenizer
from config import config


class TextDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__()

        self.tokenized_lines = []

        # Tokenize text file lines
        with open(config["ds_path"], "r") as file:
            lines = file.readlines()

            for line in lines:
                # Tokenize
                normalized_line = tokenizer.normalize(line)  # Normalize line
                tokens = tokenizer.tokenize(normalized_line)  # Tokenize normalized line
                tokenizer.build_vocab(tokens)  # Add to vocabulary with line tokens
                token_ids = tokenizer.encode(tokens)  # Get token ids from tokens

                # Keep tokenized_line
                self.tokenized_lines.append(token_ids)

            self.lines = lines

    def __len__(self):
        return len(self.lines)

    # Returns (input_ids, target_ids)
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:

        max_seq_len = config["max_seq_len"]

        # Read in token ids
        token_ids: list[int] = self.tokenized_lines[index]

        # Truncate to max_seq_len + 1 (see example for why)
        token_ids = token_ids[: max_seq_len + 1]

        # Add {pad_len} padding tokens to reach max_seq_len
        pad_len = max_seq_len + 1 - len(token_ids)
        token_ids = token_ids + [config["pad_i"]] * pad_len

        # Build target, input tensors
        input_ids = torch.tensor(token_ids[:-1])  # e.g. [A, B]
        target_ids = torch.tensor(token_ids[1:])  # e.g. [B, C]

        # If    max_seq_len = 2,
        #       token_ids = [A, B, C]   -> 3 tokens (?!)

        # then  input_ids = [A, B],     -> 2 tokens
        #       target_ids = [B, C]     -> 2 tokens :)

        return input_ids, target_ids
