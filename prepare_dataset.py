import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from tokenizer import RegexTokenizer
from config import config, get_dataset_path, get_merges_path

print("Preparing dataset...")

CHUNK_SIZE = 1024  # Number of processed examples between file writes

ds = load_dataset("Skylion007/openwebtext", streaming=True, split="train")

assert ds.info.splits
num_examples = ds.info.splits["train"].num_examples


tok = RegexTokenizer()
tok.register_pattern(config["tokenizer_pattern"])
tok.register_special_tokens(config["special_tokens"])

# Load tokenizer model state from disk
assert os.path.exists(get_merges_path()), "Missing tokenizer model!"

print("Loading saved tokenizer...")
tok.load()


token_ids = []
with open(get_dataset_path(), "wb") as f:

    for i, example in tqdm(enumerate(ds), total=num_examples):
        text = example["text"]
        token_ids.extend(tok.encode(text, allowed_special="all"))
        token_ids.append(config["special_tokens"]["<|endoftext|>"])

        if i % CHUNK_SIZE == 0:
            arr = np.array(token_ids, dtype=np.uint16)
            token_ids = []
            arr.tofile(f)
