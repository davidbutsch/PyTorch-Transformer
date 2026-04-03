import os
import json
import re
from config import config


class Tokenizer:
    def __init__(self) -> None:
        # Default vocab tokens
        self.vocabs = {"<PAD>": config["pad_i"], "<UNK>": 1}
        self.inverse_vocab = {config["pad_i"]: "<PAD>", 1: "<UNK>"}

        # Load vocab from disk
        if os.path.exists(config["vocabs_path"]):
            with open(config["vocabs_path"], "r") as file:
                print("Loading saved vocabulary...")
                self.vocabs = json.load(file)

    # Only return lowercase alphanumeric characteres
    def normalize(self, text: str) -> str:
        alphanumeric = re.sub(r"[^a-zA-Z0-9 ]", "", text)
        return alphanumeric.lower()

    # Split input text into seperate tokens
    def tokenize(self, text: str) -> list[str]:
        return text.split()

    # Add each new unique token to set and assign token id
    def build_vocab(self, tokens: list[str]):
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token not in self.vocabs:
                # Insert new vocab token into both maps
                self.vocabs[token] = len(self.vocabs)
                self.inverse_vocab[len(self.inverse_vocab)] = token

    # Encode list of tokens to list of token ids
    def encode(self, tokens) -> list[int]:
        return [self.vocabs.get(token, self.vocabs["<UNK>"]) for token in tokens]

    # Decode list of token ids to list of tokens
    def decode(self, token_ids) -> list[str]:

        return [self.inverse_vocab.get(token_id, "<UNK>") for token_id in token_ids]
