import huggingface_hub
import multiprocessing
from itertools import chain
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

# load the bookcorpus dataset
dataset = load_dataset('bookcorpus')

# split the data into train split
split_dataset = dataset['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['train_bookcorpus'] = split_dataset.pop('train') # rename the train split to train_bookcorpus
split_dataset['val_bookcorpus'] = split_dataset.pop('test') # rename the test split to val_bookcorpus

# load the bart tokenizer from facebook
enc = BartTokenizerFast.from_pretrained('facebook/bart-base')

# define the preprocessing functions˙
def process(examples):
    return enc(
       examples["text"], return_special_tokens_mask=True, truncation=True, max_length=512
    )

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= enc.model_max_length:
        total_length = (total_length // 512) * 512
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + 512] for i in range(0, total_length, 512)]
        for k, t in concatenated_examples.items()
    }
    return result

# apply the process to the datasets and shuffle
tokenized_datasets = split_dataset.map(process, batched=True, remove_columns=["text"], num_proc=num_proc)
tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
tokenized_datasets = tokenized_datasets.shuffle(seed=34)

# push the preprocess data into the hub
dataset_id = f"{user_id}\processed_bart_data"
tokenized_datasets.push_to_hub("processed_bart_data")