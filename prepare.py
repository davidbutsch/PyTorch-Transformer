import os
import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from tokenizer import RegexTokenizer
from config import config, get_merges_path

# Number of tokens per shard file — adjust based on available disk space
SHARD_SIZE = int(1e8)  # 100M tokens


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data", config["dataset_prefix"])
    os.makedirs(data_dir, exist_ok=True)

    tok = RegexTokenizer()
    tok.register_pattern(config["tokenizer_pattern"])
    tok.register_special_tokens(config["special_tokens"])
    assert os.path.exists(
        get_merges_path()
    ), "Missing tokenizer file. Run run train_tokenizer.py to train the tokenizer first."
    tok.load()

    eot = config["special_tokens"]["<|endoftext|>"]
    ds = load_dataset("Skylion007/openwebtext", streaming=True, split="train")

    shard_idx = 0
    shard_tokens: list[int] = []
    val_written = False

    def flush_shard(tokens: list[int], path: str):
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(path)
        print(f"  wrote {len(arr):,} tokens → {path}")

    for example in tqdm(ds, desc="tokenizing"):
        doc_tokens = tok.encode(example["text"], allowed_special="all")
        doc_tokens.append(eot)
        shard_tokens.extend(doc_tokens)

        # Drain full shards as they fill up
        while len(shard_tokens) >= SHARD_SIZE:
            chunk = shard_tokens[:SHARD_SIZE]
            shard_tokens = shard_tokens[SHARD_SIZE:]

            if not val_written:
                # Reserve first full shard as validation data
                path = os.path.join(data_dir, "val_0000.bin")
                val_written = True
            else:
                path = os.path.join(data_dir, f"train_{shard_idx:04d}.bin")
                shard_idx += 1

            flush_shard(chunk, path)

    # Write any leftover tokens as the final train shard
    if shard_tokens:
        path = os.path.join(data_dir, f"train_{shard_idx:04d}.bin")
        flush_shard(shard_tokens, path)

    meta = {
        "vocab_size": config["vocab_size"],
        "tokenizer_prefix": config["tokenizer_prefix"],
    }
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"saved meta → {meta_path}")
    print(f"done: {shard_idx} train shards + 1 val shard")


if __name__ == "__main__":
    main()
