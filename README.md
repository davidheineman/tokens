Pulls huggingface datasets and trains tokenizers on those datasets.

```sh
# for tokenization
pip install tokenizers datasets "huggingface_hub[cli]" ipykernel ipywidgets pandas pyarrow

# for training
pip install transformers torch

# for wikitext
huggingface-cli download Salesforce/wikitext --repo-type dataset --local-dir wikitext/
```