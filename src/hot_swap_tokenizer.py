from pathlib import Path
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
import torch.nn as nn

from constants import DATA_DIR

def hot_swap_tokenizer(model_name: str, tokenizer_path: Path):
    print("Loading original model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu") # we only need cpu
    orig_embed_dim = model.get_input_embeddings().embedding_dim
    
    print("Loading new tokenizer...")
    new_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    new_vocab_size = new_tokenizer.get_vocab_size()
    
    print("Creating new embedding layer...")
    new_embeddings = nn.Embedding(new_vocab_size, orig_embed_dim)
    
    nn.init.xavier_uniform_(new_embeddings.weight)
    
    print("Replacing embedding layer...")
    model.set_input_embeddings(new_embeddings)
    
    if model.get_output_embeddings() is not None:
        model.set_output_embeddings(new_embeddings)
    
    print(f"Successfully swapped tokenizer. New vocabulary size: {new_vocab_size}")
    return model, new_tokenizer


if __name__ == "__main__":
    # Example usage
    tokenizer_name = "s1"
    model_name = "allenai/OLMo-2-1124-7B"
    tokenizer_path = DATA_DIR / f"{tokenizer_name}.json"
    
    model, tokenizer = hot_swap_tokenizer(model_name, tokenizer_path)

    # Save model with new tokenizer
    output_path = DATA_DIR / f"{model_name.split('/')[-1]}-{tokenizer_name}_tokenizer"
    print(f"Saving model to {output_path}...")
    model.save_pretrained(output_path)
