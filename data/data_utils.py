import torch
from typing import List
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from prepare_tokenizers import MachineTranslationTokenizer
from torch.utils.data import Dataset, DataLoader

# Define a collator class so we avoid nested programming
class MachineTranslationCollator:
    def __init__(self, vocab_dir:str, langs:List[str]):
        self.tokenizer = MachineTranslationTokenizer(vocab_dir)
        self.src_lang = langs[0]
        self.tgt_lang = langs[1]

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for sentence in batch['translation']:
            # tensorize the input sentence
            src_batch.append(torch.tensor(self.tokenizer.encode(sentence[self.src_lang]).ids))
            tgt_batch.append(torch.tensor(self.tokenizer.encode(sentence[self.tgt_lang]).ids))
        # pad the sequences
        src_batch = pad_sequence(src_batch, padding_value=1)
        tgt_batch = pad_sequence(tgt_batch, padding_value=1)

        return {"src_ids": src_batch, "tgt_ids": tgt_batch}
    
def get_dataloaders(
        dataset,
        vocab_dir:str, 
        langs:List[str],
        batch_size:int, 
        is_distributed:bool=False
    ):
    
    collator = MachineTranslationCollator(vocab_dir, langs)
    # declare the samplers
    train_sampler = (DistributedSampler(dataset) if is_distributed else None)
    val_sampler = (DistributedSampler(dataset) if is_distributed else None)

    # declare the dataloaders
    train_dataloader = DataLoader(
        dataset['train'],
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collator.collate_fn
    )

    val_dataloader = DataLoader(
        dataset['val'],
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=val_sampler,
        collate_fn=collator.collate_fn
    )

    return train_dataloader, val_dataloader
