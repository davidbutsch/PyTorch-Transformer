import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizer import Tokenizer
from config import config, get_preprocessed_ds_path, get_vocabs_path
from datasets import load_dataset


class TextDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__()

        binary_path = get_preprocessed_ds_path()
        vocabs_path = get_vocabs_path()

        if not os.path.exists(binary_path):
            print("Preprocessing dataset...")

            ds = load_dataset(config["dataset"], split="train", streaming=True)

            with open(binary_path, "wb") as f:
                buffer = []
                buffer_size = 10_000_000  # 10M tokens before flush

                for i, example in enumerate(ds):
                    normalized = tokenizer.normalize(example["text"])
                    tokens = tokenizer.tokenize(normalized)
                    tokenizer.build_vocab(tokens)
                    token_ids = tokenizer.encode(tokens)
                    buffer.extend(token_ids)

                    # Flush after reaching buffer_size
                    if len(buffer) >= buffer_size:
                        np.array(buffer, dtype=np.uint16).tofile(f)
                        print(f"Flushed {len(buffer):,} tokens (example {i:,})")
                        buffer = []  # Clean buffer

                # Final flush
                if buffer:
                    np.array(buffer, dtype=np.uint16).tofile(f)

            print(f"Binary tokenization complete")

            # Save vocabulary to disk
            if not os.path.exists(vocabs_path):
                with open(vocabs_path, "w") as f:
                    json.dump(tokenizer.vocabs, f)
                print(f"Saved vocabulary: {len(tokenizer.vocabs)} tokens")

            print("Preprocessing complete!")

        # Memory-map the binary file
        self.tokenized_dataset = np.memmap(binary_path, dtype=np.uint16, mode="r")
        print(f"Memory-mapped {len(self.tokenized_dataset):,} tokens")

    def __len__(self):
        return len(self.tokenized_dataset) // config["max_seq_len"]

    # Returns (input_ids, target_ids)
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        # Read in token ids in range [index * max_seq_len, (index + 1) * max_seq_len + 1]
        start = index * config["max_seq_len"]
        end = (index + 1) * config["max_seq_len"] + 1
        token_ids = self.tokenized_dataset[start:end].astype(np.int64)

        # Padding if needed
        if len(token_ids) < config["max_seq_len"] + 1:
            pad_len = config["max_seq_len"] + 1 - len(token_ids)
            token_ids = np.pad(token_ids, (0, pad_len), constant_values=config["pad_i"])

        # Build target, input tensors
        input_ids = torch.from_numpy(token_ids[:-1].copy())  # e.g. [A, B]
        target_ids = torch.from_numpy(token_ids[1:].copy())  # e.g. [B, C]

        # If    max_seq_len = 2,
        #       token_ids = [A, B, C]   -> 3 tokens (?!)

        # then  input_ids = [A, B],     -> 2 tokens
        #       target_ids = [B, C]     -> 2 tokens :)

        return input_ids, target_ids
