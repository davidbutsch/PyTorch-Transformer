import os
import torch

from config import get_model_path
from generator import Generator


def main():

    # Load model state from disk
    if not os.path.exists(get_model_path()):
        print("Missing model save file!")

    print("Loading saved model...")
    state = torch.load(get_model_path())
    generator = Generator(state)

    while True:
        prompt = input("")

        response = generator.generate(prompt)

        print(response)


if __name__ == "__main__":
    main()
