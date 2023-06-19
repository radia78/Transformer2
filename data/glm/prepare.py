import huggingface_hub
import multiprocessing
from itertools import chain
from datasets import load_dataset, concatenate_datasets # huggingface datasets
from transformers import BartTokenizerFast
from huggingface_hub import HfApi

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = multiprocessing.cpu_count() // 2

# load the wikipedia and bookcorpus dataset
wiki = load_dataset('wikipedia', '20220301.en', split='train')
wiki.remove_columns([col for col in wiki.column_names if col != "text"])
bookcorpus = load_dataset('bookcorpus', split='train')

# unit test that the features are th same
assert bookcorpus.features.type == wiki.features.type

# concatentate and remove the temporary datasets
raw_datasets = concatenate_datasets([bookcorpus, wiki])
del wiki
del bookcorpus

# split the data into train split
split_dataset = raw_datasets['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['train'] = split_dataset.pop('train')
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

# delete the raw datasets from memory
del raw_datasets

# load the bart enc from facebook
enc = BartTokenizerFast.from_pretrained('facebook/bart-base', model_max_length=512)

def process(examples):
    tokenized_inputs = enc(
       examples["text"], return_special_tokens_mask=True, truncation=True, max_length=enc.model_max_length
    )
    return tokenized_inputs

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= enc.model_max_length:
        total_length = (total_length // enc.model_max_length) * enc.model_max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + enc.model_max_length] for i in range(0, total_length, enc.model_max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

# preprocess dataset
tokenized_datasets = raw_datasets.map(process, batched=True, remove_columns=["text"], num_proc=num_proc)
tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)

# shuffle the dataset
tokenized_datasets = tokenized_datasets.shuffle(seed=34)

if __name__ == "__main__":
    # logging into huggingfacehub using the access tokens
    access_tokens = "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"
    huggingface_hub.login(access_tokens)

    # get the user id for this run
    user_id = HfApi().whoami()["name"]
    print(f"user id '{user_id}' will be used during the example")

    # push the preprocess data into the hub
    dataset_id = f"{user_id}\glm_pretraining_data"
    tokenized_datasets.push_to_hub("glm_pretraining_data")
