import torch

from config import config, get_checkpoint_path, get_model_path, get_merges_path
from generator import Generator


def main():
    # Prefer the new nanoGPT-style checkpoint; fall back to the legacy model path
    ckpt_path = get_checkpoint_path()
    if not ckpt_path.exists():
        ckpt_path = get_model_path()
    assert ckpt_path.exists(), (
        "No model checkpoint found. Run train.py to train a model first."
    )
    assert get_merges_path().exists(), "Missing tokenizer file."

    print(f"Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=config["device"], weights_only=True)
    generator = Generator(state)

    while True:
        prompt = input("Prompt: ")
        generator.generate(prompt)
        print("\n")


if __name__ == "__main__":
    main()
