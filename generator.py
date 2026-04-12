import torch
from model import Transformer
from tokenizer import Tokenizer
from config import config


class Generator:
    def __init__(self):

        # Instantiate transformer, tokenizer
        self.model = Transformer().to(config["device"])
        self.tokenizer = Tokenizer()

        # Put model in evaluation mode
        self.model.eval()

        self.context: list[int] = []

    def generate(self, prompt: str, max_new_tokens=20) -> str:

        # Tokenize
        normalized_prompt = self.tokenizer.normalize(prompt)
        tokens = self.tokenizer.tokenize(normalized_prompt)
        self.tokenizer.build_vocab(tokens)

        self.context = self.tokenizer.encode(tokens)

        response_ids: list[int] = []

        # Autoregressive loop
        for _ in range(max_new_tokens):

            logits: torch.Tensor = self.model(
                torch.tensor([self.context], dtype=torch.int, device=config["device"])
            )  # (batch, seq_len, vocab_size)

            # Greedy sample from last row
            next_token_id = int(torch.argmax(logits[:, -1, :]).item())

            # Add prediction to conversation context and response_ids
            self.context.append(next_token_id)
            response_ids.append(next_token_id)

        # Decode token_ids to text
        tokens = self.tokenizer.decode(response_ids)
        response = "".join(tokens)

        return response
