Train tokenizers on HF datasets and hot-swap the tokenizer of a model.

### Setup

```sh
# for tokenization
pip install tokenizers datasets "huggingface_hub[cli]" ipykernel ipywidgets pandas pyarrow

# for training
pip install transformers torch

# for adaptation
git clone -b davidh/tokenizer https://github.com/allenai/open-instruct
pip install -e open-instruct/.
```

### Train new tokenizers
```sh
# download wikitext
huggingface-cli download Salesforce/wikitext --repo-type dataset --local-dir wikitext/

# download data
python src/pull_data.py

# train new tokenizers
python src/train_tokenizer.py
```

### SFT 7B

```sh
cd open-instruct

./scripts/train/olmo2/finetune_7b.sh
```