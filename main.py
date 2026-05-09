import os
import torch

from config import config, get_model_path, get_merges_path
from generator import Generator


def main():

    # Load model state from disk
    assert os.path.exists(get_model_path()), "Missing model file."
    assert os.path.exists(get_merges_path()), "Missing tokenizer file"

    print("Loading saved model...")
    state = torch.load(
        get_model_path(), map_location=config["device"], weights_only=True
    )
    generator = Generator(state)

    while True:
        prompt = input("Prompt: ")

        generator.generate(prompt)
        print("\n")


if __name__ == "__main__":
    main()
