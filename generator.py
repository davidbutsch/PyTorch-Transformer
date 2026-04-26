import torch
from model import Transformer
from tokenizer import Tokenizer
from config import config


class Generator:
    def __init__(self, state):

        # Instantiate transformer, tokenizer
        self.model = Transformer(state).to(config["device"])
        self.tokenizer = Tokenizer()

        # Put model in evaluation mode
        self.model.eval()

        self.context: list[int] = []

    def generate(self, prompt: str, max_new_tokens=200, temperature=1.0) -> str:

        # Tokenize
        normalized_prompt = self.tokenizer.normalize(prompt)
        tokens = self.tokenizer.tokenize(normalized_prompt)

        self.context = self.tokenizer.encode(tokens)

        response_ids: list[int] = []

        # Autoregressive loop
        for _ in range(max_new_tokens):

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

            # Add prediction to conversation context and response_ids
            self.context.append(next_token_id)
            response_ids.append(next_token_id)

        # Decode token_ids to text
        tokens = self.tokenizer.decode(response_ids)
        response = "".join(tokens)

        return response
