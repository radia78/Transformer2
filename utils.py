import torch
import os
import math
import inspect
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from multiprocessing import cpu_count
from torch.utils.data import DistributedSampler

# Define a collator class so we avoid nested programming
class MachineTranslationCollator:
    def __init__(self, tokenizer, langs, max_len, pad_idx: int=1):
        self.tokenizer = tokenizer
        self.pad_idx = pad_idx
        self.src_lang = langs[0]
        self.tgt_lang = langs[1]
        self.max_len = max_len

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for sentence in batch:
            # tensorize the input sentence
            src_batch.append(torch.tensor(self.tokenizer.encode(sentence[self.src_lang], truncation=True, max_length=self.max_len)))
            tgt_batch.append(torch.tensor(self.tokenizer.encode(sentence[self.tgt_lang], truncation=True, max_length=self.max_len)))
        # pad the sequences
        src_batch = pad_sequence(src_batch, padding_value=self.pad_idx, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.pad_idx, batch_first=True)

        # source batch, target batch
        return src_batch, tgt_batch

# load the data and send it across nodes/GPUs
def get_data(batch_size, langs, max_len, distributed=False):
    wmt14 = load_dataset("wmt14", f"{langs[0]}-{langs[1]}", split="train")['translation'] # load the data from huggingface
    tokenizer = PreTrainedTokenizerFast.from_pretrained('radia/wmt14-de2en-tokenizer') # load the pretrained tokenizer from huggingface
    collator = MachineTranslationCollator(tokenizer, langs, max_len) # create the collating function

    num_workers = cpu_count() // 2 # find the number of cores that can work on 
    sampler = DistributedSampler(wmt14) if distributed else None # this sampler is important so it sends data to GPUs and nodes

    dataloader = DataLoader(
        dataset=wmt14,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collator.collate_fn,
        shuffle=False if distributed else True,
        num_workers=num_workers
    )

    return dataloader

def LRDecayWarmup(iter:int, warmup_iters: int, dim:int, lr: float):
    return dim**-0.5 * min(iter**-0.5, iter * warmup_iters ** -1.5)

def get_lr_scheduler(optimizer, warmup_iters: int, dim: int, lr: float):
    # create the linear warmup and cosine decay
    lr_lambda = lambda iter: LRDecayWarmup(iter, warmup_iters, dim, lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler

# function to implement weight decay to only parameters that have a higher dimension
def configure_optimizer(model, learning_rate, betas, eps, device_type):
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

# setup the logging directories
def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)

if __name__ == "__main__":
    dl = get_data(32, ['de', 'en'], 512)
    src_sample, tgt_sample = next(iter(dl))
    print(src_sample.shape)
    print(tgt_sample.shape)