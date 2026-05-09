import time
import torch
from datasets import IterableDataset, load_dataset

from config import config
from tokenizer.bpe_tokenizer import RegexTokenizer


class PackedDataset(IterableDataset):

    def __init__(self, tokenizer: RegexTokenizer, batch_size, max_seq_len) -> None:
        self.tokenizer = tokenizer
        self.tokenizer.register_special_tokens(config["special_tokens"])

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.dataset = load_dataset(config["dataset"], split="train", streaming=True)
        # Sample examples from a chunk of size `buffer_size`
        # Seeed with random int in 2^32
        self.dataset.shuffle(
            buffer_size=10_000,
            seed=hash(f"{time.time()}") % 2**32,
        )

    def __iter__(self):
        # chunk_size used to dump token_buffer when capacity reached (+1 token per batch for input/target offset)
        chunk_size = self.batch_size * (self.max_seq_len + 1)

        token_buffer: list[int] = []
        for example in self.dataset:
            tokens = self.tokenizer.encode(example["text"], allowed_special="all")
            token_buffer.extend(tokens)

            # Add end-of-text token to end of each example
            token_buffer.append(config["special_tokens"]["<|endoftext|>"])

            # Yield buffer when capacity reached (chunk_size)
            while len(token_buffer) >= chunk_size:

                # Get chunk from buffer
                # (batch_size, max_seq_len + 1)
                chunk = torch.tensor(token_buffer[:chunk_size]).view(
                    self.batch_size, self.max_seq_len + 1
                )

                # Slice buffer and omit chunk contents
                token_buffer = token_buffer[chunk_size:]

                # (batch_size, max_seq_len)
                input_ids = chunk[:, :-1]  # end at -1
                target_ids = chunk[:, 1:]  # start at 1

                yield input_ids, target_ids
