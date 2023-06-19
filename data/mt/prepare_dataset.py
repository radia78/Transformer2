import huggingface_hub
import multiprocessing
from datasets import load_dataset
from huggingface_hub import HfApi

# load the data and the number of workers spared
num_proc = multiprocessing.cpu_count() // 2
ds = load_dataset("wmt14", "de-en")

# create an average length column
def create_ln(examples):
    len1 = len(examples['translation']['de'])
    len2 = len(examples['translation']['en'])
    avg_len = (len1 + len2)/2

    return {"translation": examples['translation'], "len": avg_len}

ds = ds.map(create_ln) # map the process to the dataset
ds = ds = ds.sort(['len'], reverse=False)

if __name__=="__main__":
    # logging into huggingfacehub using the access tokens
    access_tokens = "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"
    huggingface_hub.login(access_tokens)
    user_id = HfApi().whoami()["name"] # get the user id for this run

    # push the preprocess data into the hub
    dataset_id = f"{user_id}\wmt14-cleaned"
    ds.push_to_hub(dataset_id)
