import pandas as pd
import os
import glob
from datasets import load_dataset, Dataset, DatasetDict

from pathlib import Path
from constants import DATA_DIR

def convert_parquet_to_txt(dataset_name, text_key="text"):
    text = ""
    pattern = f"wikitext/wikitext-103-raw-v1/{dataset_name}-*-of-*.parquet"
    parquet_files = glob.glob(pattern)

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        text += "".join(df[text_key])

    os.makedirs("wikitext/wikitext-103-raw-v1-txt", exist_ok=True)
    output_path = f"wikitext/wikitext-103-raw-v1-txt/{dataset_name}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def split_and_save_ds(ds: Dataset, ds_name: Path, text_key="text"):
    # Split into train/val/test
    ds: DatasetDict = ds.train_test_split(test_size=0.2, seed=42)
    test_valid: DatasetDict = ds["test"].train_test_split(test_size=0.5, seed=42)

    train_ds = ds["train"]
    valid_ds = test_valid["train"]
    test_ds = test_valid["test"]

    # Write to txt files
    os.makedirs(ds_name, exist_ok=True)

    with open(ds_name / "train.txt", "w", encoding="utf-8") as f:
        f.write("".join(train_ds[text_key]))

    with open(ds_name / "validation.txt", "w", encoding="utf-8") as f:
        f.write("".join(valid_ds[text_key]))

    with open(ds_name / "test.txt", "w", encoding="utf-8") as f:
        f.write("".join(test_ds[text_key]))


def create_s1_text(entry):
    text = (
        entry["question"]
        + "\n"
        + entry["gemini_thinking_trajectory"]
        + entry["gemini_attempt"]
    )
    return {"text": text}


def create_openthoughts_text(entry):
    text = ""
    for message in entry["conversations"]:
        text += message["value"]
    return {"text": text}


def main():
    # Process wikitext
    convert_parquet_to_txt("train")
    convert_parquet_to_txt("validation")
    convert_parquet_to_txt("test")

    # Process s1 dataset
    ds = load_dataset("simplescaling/s1K-1.1")["train"]
    ds = ds.map(create_s1_text)
    split_and_save_ds(ds, DATA_DIR / "s1")

    # Process OpenThoughts dataset
    ds = load_dataset("open-thoughts/OpenThoughts3-1.2M")["train"]
    ds = ds.map(create_openthoughts_text)
    split_and_save_ds(ds, DATA_DIR / "open_thoughts")


if __name__ == "__main__":
    main()
