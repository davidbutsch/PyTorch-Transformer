import torch

from config import config, get_checkpoint_path, get_merges_path
from generator import Generator


def main():
    ckpt_path = get_checkpoint_path()
    assert (
        ckpt_path.exists()
    ), "No model checkpoint found. Run train.py to train a model first."
    assert (
        get_merges_path().exists()
    ), "Missing tokenizer file. Run train_tokenizer.py to train a tokenizer first."

    print(f"Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=config["device"], weights_only=True)
    generator = Generator(state)

    while True:
        prompt = input("Prompt: ")
        generator.generate(prompt)
        print("\n")


if __name__ == "__main__":
    main()
