import torch
from model import Transformer
from tokenizer import RegexTokenizer
from config import config


class Generator:
    def __init__(self, state):

        # Instantiate transformer, tokenizer
        self.model = Transformer(state).to(config["device"])
        self.tokenizer = RegexTokenizer()
        self.tokenizer.load()
        self.tokenizer.register_special_tokens(config["special_tokens"])

        # Put model in evaluation mode
        self.model.eval()

        self.context: list[int] = []

    def generate(self, prompt: str, temperature=0.5) -> None:

        # Tokenize
        token_ids = self.tokenizer.encode(prompt.strip())

        self.context: list[int] = token_ids

        response_ids: list[int] = []

        # Autoregressive loop
        while True:

            logits: torch.Tensor = self.model(
                torch.tensor([self.context], dtype=torch.int, device=config["device"])
            )  # (batch, seq_len, vocab_size)

            # Pull out last token (prediction)
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

            # Convert to probability distribution
            next_token_probs = torch.softmax(next_token_logits / temperature, dim=-1)

            # Sample from next token probability
            next_token_id = int(
                torch.multinomial(next_token_probs, num_samples=1).item()
            )

            # Add prediction to conversation context and print
            self.context.append(next_token_id)
            response_ids.append(next_token_id)

            # Decode token_ids to text
            new_token = self.tokenizer.decode(response_ids[-1:])

            print(new_token, end="", flush=True)

            if next_token_id == config["special_tokens"]["<|endoftext|>"]:
                print(f"\nNEXT_TOKEN_ID!!!... len(response_ids) = {len(response_ids)}")

            if (
                next_token_id == config["special_tokens"]["<|endoftext|>"]
                or len(response_ids) > 500
            ):
                break
