import torch
import os
import inspect
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from multiprocessing import cpu_count
from model import Transformer
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
    return dim**-0.5 * min((iter + 1)**-0.5, (iter + 1) * warmup_iters ** -1.5)

def get_lr_scheduler(optimizer, warmup_iters: int, dim: int, lr: float):
    # create the linear warmup and cosine decay
    lr_lambda = lambda iter: LRDecayWarmup(iter, warmup_iters, dim, lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler

# function to implement weight decay to only parameters that have a higher dimension
def configure_optimizer(model, learning_rate, betas, device_type):
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

# generating function
class Generator:
    def __init__(self, ckpt_path, model_config, device):
        # load the model
        self.model = self.load_model(ckpt_path, model_config, device)
        self.model.eval()

        # initialize tokenizer
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('radia/wmt14-de2en-tokenizer')

        self.device = device

    def load_model(self, ckpt_path, model_config, device):
        model = Transformer(model_config)
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
        model.to(device)
        return model
    
    def decode(self, src_sent, max_len, temperature, k, decode_method='greedy'):

        # send the src tokens to device
        src_tokens = self.tokenizer(src_sent, return_tensors='pt', max_length=max_len, truncation=True)

        if decode_method == 'greedy':
            results = self.greedy_search(src_tokens.input_ids, max_len, temperature)
            results = self.tokenizer.decode(results.squeeze(0), skip_special_tokens=True)
        
        if decode_method == 'beam':
            candidates = self.beam_search(src_tokens.input_ids, max_len, k, temperature)
            results = self.tokenizer.decode(candidates[0][-1].squeeze(0))

        return results
    
    def greedy_search(self, src_tokens, max_len, temperature=1.0):

        # cache the number of batches
        b = src_tokens.shape[0]
        # create a target dummy with only the start of sentence token
        idx = torch.zeros(b, 1, device=self.device).long()
        # send the source tokens into the same device
        src_tokens = src_tokens.to(self.device)

        for _ in range(max_len):

            # forward the model to get the logits for the index in the sequence
            with torch.no_grad():
                logits = self.model(src_tokens, idx)
                logits = logits[:, -1, :] / temperature

                if b == 1:
                    logits = logits.unsqueeze(0)

            # choose the most likely guest
            idx_next = logits.argmax(2)
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            # break the process if idx_next is the eos_token
            if idx_next == 2:
                break

        return idx.detach()

    def beam_search(self, src_tokens, max_len, k=5, temperature=1.0):
        # cache the number of batches
        b = src_tokens.shape[0]

        # create a list of candidate tensors
        candidates = []

        # create a temporary cache for hypothesis
        hypotheses = [(0, torch.zeros(b, 1, device=self.device).long())]

        # loop through the maximum sequence length
        for n in range(max_len):

            # loop through all current hypotheses
            for i in range(len(hypotheses)):

                # forward the model to get the logits for the index in the sequence
                with torch.no_grad():
                    logits = self.model(src_tokens, hypotheses[i][1])
                    logits = logits[:, -1, :] / temperature

                # choose the top 4 candidate
                scores, indices = logits.topk(k, dim=-1, largest=True, sorted=True)
                scores += hypotheses[i][0]

                if b == 1:
                    indices = indices.unsqueeze(0)

                for j in range(k):
                    # append candidates
                    hypotheses.append(
                        (scores[0][j].item(), torch.cat((hypotheses[i][1], indices[:, 0, j].unsqueeze(0)), dim=-1))
                    )

            # pop out the initial value
            if n == 0:
                hypotheses = hypotheses[1:]

            # pruning the list
            hypotheses.sort(key=lambda x: x[0], reverse=True)
            hypotheses = hypotheses[:4]

            # add the hypotheses list to candidates
            candidates += hypotheses

        # normalize the score
        candidates = list(map(lambda x: (x[0]/x[1].shape[-1]**0.6, x[1]), candidates))
        candidates = sorted(candidates, reverse=True)

        return candidates

# setup the logging directories
def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)

if __name__ == "__main__":
    # testing the data sample
    dl = get_data(32, ['de', 'en'], 512)
    src_sample, tgt_sample = next(iter(dl))
    print(src_sample.shape)
    print(tgt_sample.shape)