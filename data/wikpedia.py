import huggingface_hub
import multiprocessing
from datasets import load_dataset # huggingface datasets
from transformers import BartTokenizerFast
from huggingface_hub import HfApi

# logging into huggingfacehub using the access tokens
access_tokens = "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"
huggingface_hub.login(access_tokens)

# get the user id for this run
user_id = HfApi().whoami()["name"]
print(f"user id '{user_id}' will be used during the example")

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = multiprocessing.cpu_count() // 2

# load the wikipedia dataset
dataset = load_dataset('wikipedia')

# split the data into train split
split_dataset = dataset['train'].train_twest_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['train_wikipedia'] = split_dataset.pop('train')
split_dataset['val_wikipedia'] = split_dataset.pop('test') # rename the test split to val

# load the bart tokenizer from facebook
enc = BartTokenizerFast.from_pretrained('facebook/bart-base')

# define the preprocessing function
def process(examples):
    return enc(
       examples["text"], return_special_tokens_mask=True, truncation=True, max_length=512
    )

# apply the process to the datasets and shuffle
tokenized_datasets = split_dataset.map(process, batched=True, remove_columns=["text"], num_proc=num_proc)
tokenized_datasets = tokenized_datasets.shuffle(42)

# push the preprocess data into the hub
dataset_id = f"{user_id}\processed_bart_data"
tokenized_datasets.push_to_hub("processed_bart_data")