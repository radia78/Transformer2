import re
import huggingface_hub
from typing import List
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from html2text import html2text
from huggingface_hub import HfApi

# helper functions
def clean_punctuation(text):
    pattern1 = r"\s+([?.!;:,])" # clean the spacing between sentence punctuations
    pattern2 = r"(?<=\d)\s*:\s*(?=\d)" # clean colons with numbers
    pattern3 = r"\s*([/\\'])\s*" # clean spaces between slashes and apostrophes
    pattern4 = r"([\(\[\{])\s*(.*?)\s*([\)\]\}])" # clean the spaces inside
    result = re.sub(pattern1, r'\1', text)
    result = re.sub(pattern2, r':', result)
    result = re.sub(pattern3, r'\1', result)
    result = re.sub(pattern4, r'\1\2\3', result)
    return result

def replace_compound_words(text):
    pattern1 = r'\s*##AT##-##AT##\s*'
    pattern2 = r'\s*##UNDERSCORE##\s*'
    pattern3 = r'\s*##STAR##\s*'
    result = re.sub(pattern1, r'-', text)
    result = re.sub(pattern2, r'_', result)
    result = re.sub(pattern3, r'\*', result)
    return result

def apply_cleaning_process(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def gen(shards, langs):
    processing_function = apply_cleaning_process(
        html2text, 
        replace_compound_words, 
        clean_punctuation,
        lambda x: x.strip()
    ) # declare the text-processing function
    for shard in shards:
        with open(f"{shard}.{langs[0]}") as lang1, open(f"{shard}.{langs[1]}") as lang2:
            for line1, line2 in zip(lang1, lang2):
                yield {langs[0]: processing_function(line1), langs[1]: processing_function(line2)}

# create if you large memory
def create_map_style_dataset(
        file_dir:str, 
        langs:List[str]
    ):
    # declare the shards for each split
    train_shards = [f"{file_dir}/train/train"]
    val_shards = [f"{file_dir}/val/newstest{year}" for year in range(2012, 2014)]
    test_shards = [f"{file_dir}/test/newstest2014"]

    # create the map-style dataset
    ds = DatasetDict({
        "train": Dataset.from_generator(gen, gen_kwargs={"shards": train_shards, "langs": langs}),
        "val": Dataset.from_generator(gen, gen_kwargs={"shards": val_shards, "langs": langs}),
        "test": Dataset.from_generator(gen, gen_kwargs={"shards": test_shards, "langs": langs})
    })

    return ds

# create if you have small memory
def create_iterable_style_dataset(
        file_dir:str, 
        langs:List[str]
    ):
    # declare the shards for each split
    train_shards = [f"{file_dir}/train/train"]
    val_shards = [f"{file_dir}/val/newstest{year}" for year in range(2012, 2014)]
    test_shards = [f"{file_dir}/test/newstest2014"]

    # create the iterable-style dataset (cannot push to hub)
    ds = IterableDatasetDict({
        "train": IterableDataset.from_generator(gen, gen_kwargs={"shards": train_shards, "langs": langs}),
        "val": IterableDataset.from_generator(gen, gen_kwargs={"shards": val_shards, "langs": langs}),
        "test": IterableDataset.from_generator(gen, gen_kwargs={"shards": test_shards, "langs": langs})
    })

    return ds

if __name__ == "__main__":
    # logging into huggingfacehub using the access tokens
    access_tokens = "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"
    huggingface_hub.login(access_tokens)
    user_id = HfApi().whoami()["name"]
    # push the dataset into huggingface
    create_map_style_dataset("/Users/radia/Projects/ml_projects/MiniBart/data/mt/wmt14_training", ['de', 'en']).push_to_hub(f"{user_id}/wmt14-de2en")
    