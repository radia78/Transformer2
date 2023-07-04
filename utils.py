import torch
from typing import List
from torch.utils.data import DataLoader 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

# Define a collator class so we avoid nested programming
class MachineTranslationCollator:
    def __init__(self, tokenizer, langs:List[str], pad_idx: int=1):
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

        # source batch, target batch, number of tokens (exclude pad values)
        return src_batch, tgt_batch, (tgt_batch != self.pad_idx).data.sum()

# rate scheduler
def rate(step:int, n_embd:int, factor:float=1.0, warmup:int=4000):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        n_embd ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
