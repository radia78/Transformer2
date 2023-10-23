# import utilities
import os
import logging
from tqdm import tqdm
from utils import *
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter

# import torch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *

# import distributed learning stuff
import torch.multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

LOCAL_RANK = int(os.environ['LOCAL_RANK']) # local GPU id
WORLD_SIZE = int(os.environ['WORLD_SIZE']) # the number of GPUs in total
DEVICE = f'cuda:{LOCAL_RANK}'
torch.cuda.set_device(DEVICE)

def create_model(args):

    # creating the configuration of the transformers, adjustable by the user
    model_config = TransformerConfig(
        n_emb=args.n_emb,
        n_head=args.n_head,
        n_layer=args.n_layer,
        max_len=args.max_len,
        vocab_size=args.vocab_size,
        dropout=args.dropout
    )
    model = Transformer(model_config).to(LOCAL_RANK)
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK) # send the model across GPU/nodes
    return model

def train(args):

    # setup torch training computational stuff
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # setting up tensorfloat point calculations
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype, enabled=args.amp)

    # setup the training by obtaining the necessary ingredients
    setup_logging(args.run_name)
    dataloader = get_data(args.batch_size)
    model = create_model(args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # configure the optimizer
    optimizer = configure_optimizer(model, args.weight_decay, args.max_lr, args.betas, args.eps, 'cuda')

    # configure the learning rate scheduler (cosine decay with linear warmup) 
    scheduler = get_lr_scheduler(optimizer, args.warmup_iters, args.decay_iters, args.min_lr, args.max_lr)

    # configure the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    if LOCAL_RANK == 0:
        logger = SummaryWriter(os.path.join("runs", args.run_name))

    l = len(dataloader)
    model.train() # turn on the training mode for the model

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch + 1} on GPU {LOCAL_RANK}:")
        dataloader.sampler.set_epoch(epoch) # setting the shuffler for each GPU
        print(f"Epoch {epoch + 1} on GPU {LOCAL_RANK}: ")
        pbar = tqdm(dataloader)
        for i, (src, tgt) in enumerate(pbar):

            # send both to the GPU
            src = src.to(LOCAL_RANK)
            tgt = tgt.to(LOCAL_RANK)

            # compute the prediction and calculate the loss
            # use auto mixed precision
            with ctx:
                output = model(src, tgt)
                loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
                loss = loss / args.grad_accumulation_steps

            # scale the loss
            scaler.scale(loss).backward()

            if ((i + 1) % args.grad_accumulation_steps == 0) or ((i + 1) == l):
                scaler.unscale_(optimizer) # unscale the optimizer for gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip the gradient so it doesn't explode or vanish
                scaler.step(optimizer) # update the optimizer - should skip steps with NaN's or infs
                scaler.update() # update the scale factor
                optimizer.zero_grad(set_to_none=True) # zero out the gradients

                # update the learning rate
                scheduler.step()
            
            # update the progress bar   
            pbar.set_postfix(Loss=loss.item())
            logger.add_scalar("Loss", loss.item(), global_step=epoch * l + i)
    
    if (LOCAL_RANK == 0):
        torch.save(model.module.state_dict(), os.path.join("models", args.run_name, f"chkpt.pt"))

def main(args):
    init_process_group(args.backend)
    train(args)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    # setup the arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # model config arguments
    args.n_emb = 512
    args.n_head = 8
    args.n_layer = 6
    args.max_len = 512
    args.vocab_size = 37467
    args.dropout = 0.1

    # training config arguments
    args.epochs = 16
    args.run_name = "TransformerV2"
    args.amp = False
    args.seed = 13332
    args.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    args.batch_size = 16
    args.grad_accumulation_steps = 32
    args.weight_decay = 0.1
    args.max_lr = 4e-4
    args.min_lr = 4e-5
    args.betas = (0.9, 0.98)
    args.eps = 1e-6
    args.warmup_iters = int(6e3)
    args.decay_iters = int(12e4)
    args.backend='nccl'
    
    main(args)