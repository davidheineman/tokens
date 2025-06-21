Pulls huggingface datasets and trains tokenizers on those datasets.

### Setup

```sh
# for tokenization
pip install tokenizers datasets "huggingface_hub[cli]" ipykernel ipywidgets pandas pyarrow

# for training
pip install transformers torch
```

### Train tokenizer
```sh
# download wikitext
huggingface-cli download Salesforce/wikitext --repo-type dataset --local-dir wikitext/
```

### SFT 7B

```sh
cd open-instruct

./scripts/train/olmo2/finetune_7b.sh
```