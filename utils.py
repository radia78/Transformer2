import torch
import os
import inspect
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from multiprocessing import cpu_count
from torch.utils.data import DistributedSampler

# Define a collator class so we avoid nested programming
class MachineTranslationCollator:
    def __init__(self, tokenizer, langs, pad_idx: int=1):
        self.tokenizer = tokenizer
        self.pad_idx = pad_idx
        self.src_lang = langs[0]
        self.tgt_lang = langs[1]

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for sentence in batch:
            # tensorize the input sentence
            src_batch.append(torch.tensor(self.tokenizer.encode(sentence[self.src_lang])))
            tgt_batch.append(torch.tensor(self.tokenizer.encode(sentence[self.tgt_lang])))
        # pad the sequences
        src_batch = pad_sequence(src_batch, padding_value=self.pad_idx, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.pad_idx, batch_first=True)

        # source batch, target batch
        return src_batch, tgt_batch

# load the data and send it across nodes/GPUs
def get_data(batch_size):
    wmt14 = load_dataset('radia/wmt14-de2en') # load the data from huggingface
    tokenizer = PreTrainedTokenizerFast.from_pretrained('radia/wmt14-de2en-tokenizer') # load the pretrained tokenizer from huggingface
    collator = MachineTranslationCollator(tokenizer, ['de', 'en']) # create the collating function

    num_workers = cpu_count() // 2 # find the number of cores that can work on 
    sampler = DistributedSampler(wmt14['train']) # this sampler is important so it sends data to GPUs and nodes

    dataloader = DataLoader(
        dataset=wmt14['train'],
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collator.collate_fn,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader

# learning rate decay scheduler (cosine with warmup)
def lr_schedule(step, lr, warmup_steps, decay_steps):
    # cache the coefficients for the steps
    warmup_coefficient = lr/warmup_steps
    decay_coefficient = lr/decay_steps
    if step < warmup_steps:
        return step * warmup_coefficient
    
    else:
        return lr - step * decay_coefficient

# function to implement weight decay to only parameters that have a higher dimension
def configure_optimizer(model, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

# setup the logging directories
def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)