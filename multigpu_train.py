# import the important stuff
import torch
import torch.nn.functional as F
from utils import *
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from model import RopeFormer, RopeBARTConfig

# import distributed training system
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of eac process
        world_sie: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        gpu_id,
        save_every: int
    ) -> None:
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[self.gpu_id])
    
    def _run_batch(
            self, 
            source,
            target,
            ntokens, 
            pad_idx: int=1,
            label_smoothing: float=0.1,
            reduction: str='sum'
    ):
        self.optimizer.zero_grad(set_to_none=True) # clear the gradients
        output = self.model(source, target)
        total_loss = F.cross_entropy(
            # (batch size, seq len, vocab size) -> (batch size x seq len, vocab size)
            output.contiguous().view(-1, output.size(-1)),
            # (batch size, seq len) -> (batch size x seq len)
            target.contiguous().view(-1),
            ignore_index=pad_idx, # ignore computing the padding
            reduction=reduction,
            label_smoothing=label_smoothing
        )
        norm_loss = total_loss/ntokens # norm the loss by number of tokens
        norm_loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()


    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0]) # get batch size
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_data)}")
        for source, target, ntokens in self.train_data:
            source = source.to(self.gpu_id)
            targets = target.to(self.gpu_id)
            self._run_batch(source, targets, ntokens)

    def _save_checkpoint(self, epoch):
        checkpoint = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(checkpoint, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self.model.train()
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def load_training_objects():
    ds = load_dataset('radia/wmt14-de2en') # load the dataset
    tokenizer = PreTrainedTokenizerFast.from_pretrained('radia/wmt14-de2en-tokenizer') # load tokenizer
    collate_fn = MachineTranslationCollator(tokenizer, ['de', 'en'], 1).collate_fn # load collating function

    # load model config
    model_config = RopeBARTConfig(
        vocab_size=tokenizer.vocab_size, # vocab size of the corpus
        pad_idx=tokenizer.convert_tokens_to_ids("<pad>"), # padding index
        # the base config from the transformer paper
        n_layer=3,
        n_head=4,
        n_embd=256,
        dropout=0.3,
        bias=True
    )

    model = RopeFormer(model_config) # load the model

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9) # load optimizer
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, n_embd=model_config.n_embd)
    )

    return ds, collate_fn, model, optimizer, lr_scheduler


def prepare_dataloader(dataset, collate_fn, batch_size:int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(dataset),
        collate_fn=collate_fn
    )

def main(rank, world_size, total_epochs, save_every):
    ddp_setup(rank, world_size)
    ds, collate_fn, model, optimizer, lr_scheduler = load_training_objects()
    train_dataloader = prepare_dataloader(ds['test'], collate_fn, batch_size=32)
    trainer = Trainer(
        model=model,
        train_data=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        gpu_id=rank,
        save_every=save_every
    )
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    world_size = torch.cuda.device_count()

    mp.spawn(main, args=(world_size, total_epochs, save_every), nprocs=world_size)
