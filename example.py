from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())

from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], 
    vocab_size=100_000
)

# tokenizer.train(
#     files=[
#         "wikitext/wikitext-103-raw-v1-txt/train.txt",
#         "wikitext/wikitext-103-raw-v1-txt/validation.txt",
#         "wikitext/wikitext-103-raw-v1-txt/test.txt",
#     ],
#     trainer=trainer,
# )

tokenizer.train(
    files=[
        "s1/train.txt",
        "s1/validation.txt",
        "s1/test.txt",
    ],
    trainer=trainer,
)

output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]

tokenizer.save("s1.json", pretty=True)