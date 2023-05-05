import torch
from torch.utils.tensorboard import SummaryWriter
from model import *
from utils import *
import argparse
from datetime import datetime
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

# Parse argument
parser = argparse.ArgumentParser(description='Easier implementation of the Meta MBart')
parser.add_argument('--batch-size', '-b', type=int, default=32)
parser.add_argument('--epochs', '-e', type=int, default=1000)
args = parser.parse_args()

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimizer parameters
LR = 0.001
BETAS = (0.9, 0.98)
EPS = 1e-6

# Model hyperparameters pre-defined
DMODEL = 512
FF_SIZE = 2048
NHEADS = 8
NLAYERS = 6
DROPOUT = 0.1
MAXLEN = 256

# Vocab Information
def text_transforms(tensor, maxlen):
    transforms = T.Sequential(
        T.Truncate(maxlen),
        T.ToTensor() 
        )
    return transforms(tensor)

pretrain_dataset = MiniBartPreTraining('', 'MiniBart/pretraining_datasets.pkl')

# Get vocabulary information
BOS_IDX = 0
PAD_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
VOCAB_SIZE = len(pretrain_dataset.vocab.vocab)

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transforms(src_sample, MAXLEN))
        tgt_batch.append(text_transforms(tgt_sample, MAXLEN))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

model = Seq2SeqTransformer(
    VOCAB_SIZE,
    MAXLEN,
    DMODEL,
    FF_SIZE,
    NLAYERS,
    NHEADS,
    DROPOUT,
    PAD_IDX,
    DEVICE
    ).to(DEVICE)

optimizer = Adam(model.parameters(), LR, BETAS, EPS)
lr_scheduler = LinearLR(optimizer, 0.1, 0)
criterion = CrossEntropyLoss()

def TrainingStep(
        epoch_number,
        tb_writer,
        model,
        optimizer,
        scheduler,
        criterion,
        train_dataloader
        ):
    
    running_loss = 0.
    last_loss = 0.

    # Iterate through all the training batches
    for i, data in enumerate(train_dataloader):
        # Dims of the tokens (Batch Size, Seq Len)
        src, tgt = data
        print(src.shape)
        print(tgt.shape)
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # Reset the gradients
        optimizer.zero_grad()

        # Forward propagation
        output = model(src, tgt) # (Seq Len, Batch Size, Target Vocab Size)
        output = output.reshape(-1, output.shape[2]) # (Seq Len * Batch Size, Target Vocab Size)
        target = tgt.reshape(-1) # (Seq Len * Batch Size, Target Vocab Size)

        # Compute the loss of the model
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent & scheduler learning rate update
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss/1000
            print(f'    batch {i + 1} loss: {last_loss}')
            tb_x = epoch_number * len(train_dataloader) + i + 1
            tb_writer.add_scaler('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss

if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/loss_plot_{timestamp}')

    # Turn on regularization for training
    model.train(True)
    
    for epoch in range(args.epochs):
        print(f'EPOCH {epoch + 1}:')
        avg_loss = TrainingStep(
            epoch,
            writer,
            model,
            optimizer,
            lr_scheduler,
            criterion,
            pretrain_dataloader
            )
        
        writer.flush()
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
                }
            
            save_checkpoint(checkpoint) # Save Checkpoint