```sh
pip install tokenizers
pip install -U "huggingface_hub[cli]"
pip install ipykernel

# Download wikitext
huggingface-cli download Salesforce/wikitext --repo-type dataset --local-dir wikitext/
```