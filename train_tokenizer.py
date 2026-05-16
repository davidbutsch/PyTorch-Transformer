from datasets import load_dataset

from tokenizer import RegexTokenizer
from config import config as arch_config

target_chars = 50_000_000
vocab_size = arch_config["vocab_size"]

tokenizer = RegexTokenizer()
tokenizer.register_pattern(arch_config["tokenizer_pattern"])
tokenizer.register_special_tokens(arch_config["special_tokens"])

ds = load_dataset("Skylion007/openwebtext", streaming=True, split="train")
ds = ds.shuffle(seed=42, buffer_size=100_000)

chunks: list[str] = []
cur_chars = 0

for i, example in enumerate(ds):
    chunks.append(example["text"])
    cur_chars += len(example["text"])
    if cur_chars >= target_chars:
        break

text = "".join(chunks)

print(f"Training tokenizer on {len(text)/1e6:.1f}M characters! (1M chars ~= 1Mb)")

tokenizer.train(text, vocab_size)
tokenizer.save()
