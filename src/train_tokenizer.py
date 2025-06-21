from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from constants import DATA_DIR

def train_tokenizer(dataset, vocab_size=100_000):

    tokenizer = Tokenizer(BPE())

    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], 
        vocab_size=vocab_size
    )

    tokenizer.train(
        files=[
            DATA_DIR / dataset / "train.txt",
            DATA_DIR / dataset / "validation.txt",
            DATA_DIR / dataset / "test.txt",
        ],
        trainer=trainer,
    )

    output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
    print(output.tokens)

    tokenizer.save(DATA_DIR / f"{dataset}.json", pretty=True)


if __name__ == '__main__':
    train_tokenizer("s1", vocab_size=100_000)
    train_tokenizer("open_thoughts", vocab_size=100_000)