import torch
from model import Transformer
from tokenizer import RegexTokenizer
from config import config


class Generator:
    def __init__(self, state: dict) -> None:
        # Rebuild the exact model architecture used during training.
        # model_args is saved in the checkpoint so loading works correctly
        # even if config.py has been edited since that run.
        model_args = state.get("model_args", {})
        self.model = Transformer(**model_args).to(config["device"])
        self.model.load_state_dict(state["model"])
        self.model.eval()

        if model_args:
            print(f"Loaded model: N={model_args['N']}, d_model={model_args['d_model']}, h={model_args['h']}")

        self.tokenizer = RegexTokenizer()
        self.tokenizer.load()
        self.tokenizer.register_special_tokens(config["special_tokens"])

        self.context: list[int] = []

    def generate(self, prompt: str, temperature: float = 0.5) -> None:
        self.context = self.tokenizer.encode(prompt.strip())
        response_ids: list[int] = []

        while True:
            logits: torch.Tensor = self.model(
                torch.tensor([self.context], dtype=torch.long, device=config["device"])
            )  # (1, seq_len, vocab_size)

            next_token_logits = logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = int(torch.multinomial(next_token_probs, num_samples=1).item())

            self.context.append(next_token_id)
            response_ids.append(next_token_id)

            print(self.tokenizer.decode(response_ids[-1:]), end="", flush=True)

            if (
                next_token_id == config["special_tokens"]["<|endoftext|>"]
                or len(response_ids) > 500
            ):
                break
