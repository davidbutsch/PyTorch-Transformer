from config import config
from model import Transformer
from tokenizer import Tokenizer
from training import Trainer


def train():

    model = Transformer().to(config["device"])
    tokenizer = Tokenizer()

    trainer = Trainer(model, tokenizer)

    trainer.train()


if __name__ == "__main__":
    train()
