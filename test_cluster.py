import os
import uuid
import datetime
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import urllib.request
from torch.utils.data import Dataset, DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from dask_pytorch_ddp import dispatch, results
from dask_saturn import SaturnCluster
from dask.distributed import Client
from distributed.worker import logger

os.environ['SATURN_BASE_URL'] = "https://app.community.saturnenterprise.io"
os.environ['SATURN_TOKEN'] = "server-63f7f9263fa2440096a882fd1255f9d3"

os.environ['DASK_DISTRIBUTED__WORKER__DAEMON'] = 'False' 
os.environ['SATURN__JUPYTER_SETUP_DASK_WORKSPACE'] = 'true'

with urllib.request.urlopen(
    "https://saturn-public-data.s3.us-east-2.amazonaws.com/examples/pytorch/seattle_pet_licenses_cleaned.json"
) as f:
    pet_names = json.loads(f.read().decode("utf-8"))

# Our list of characters, where * represents blank and + represents stop
characters = list("*+abcdefghijklmnopqrstuvwxyz-. ")
str_len = 8

# Formats the pet names to tensors
def format_training_data(pet_names, device=None):
    def get_substrings(in_str):
        # add the stop character to the end of the name, then generate all the partial names
        in_str = in_str + "+"
        res = [in_str[0:j] for j in range(1, len(in_str) + 1)]
        return res

    pet_names_expanded = [get_substrings(name) for name in pet_names]
    pet_names_expanded = [item for sublist in pet_names_expanded for item in sublist]
    pet_names_characters = [list(name) for name in pet_names_expanded]
    pet_names_padded = [name[-(str_len + 1) :] for name in pet_names_characters]
    pet_names_padded = [
        list((str_len + 1 - len(characters)) * "*") + characters for characters in pet_names_padded
    ]
    pet_names_numeric = [[characters.index(char) for char in name] for name in pet_names_padded]

    # the final x and y data to use for training the model. Note that the x data needs to be one-hot encoded
    if device is None:
        y = torch.tensor([name[1:] for name in pet_names_numeric])
        x = torch.tensor([name[:-1] for name in pet_names_numeric])
    else:
        y = torch.tensor([name[1:] for name in pet_names_numeric], device=device)
        x = torch.tensor([name[:-1] for name in pet_names_numeric], device=device)
    x = torch.nn.functional.one_hot(x, num_classes=len(characters)).float()
    return x, y

class OurDataset(Dataset):
    def __init__(self, pet_names, device=None):
        self.x, self.y = format_training_data(pet_names, device)
        self.permute()

    def __getitem__(self, idx):
        idx = self.permutation[idx]
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

    def permute(self):
        self.permutation = torch.randperm(len(self.x))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.lstm = nn.LSTM(
            input_size=len(characters),
            hidden_size=self.lstm_size,
            num_layers=4,
            batch_first=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(self.lstm_size, len(characters))

    def forward(self, x):
        output, state = self.lstm(x)
        logits = self.fc(output)
        return logits

def train():
    num_epochs = 25
    batch_size = 16384

    worker_rank = int(dist.get_rank())
    device = torch.device(0)

    logger.info(f"Worker {worker_rank} - beginning")

    dataset = OurDataset(pet_names, device=device)
    # the distributed sampler makes it so the samples are distributed across the different workers
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    worker_rank = int(dist.get_rank())

    # the model has to both be passed to the GPU device, then has to be wrapped in DDP so it can communicate with the other workers
    model = Model()
    model = model.to(device)
    model = DDP(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        # the logger here logs to the Dask log of each worker, for easy debugging
        logger.info(
            f"Worker {worker_rank} - {datetime.datetime.now().isoformat()} - Beginning epoch {epoch}"
        )

        # this ensures the data is reshuffled each epoch
        sampler.set_epoch(epoch)
        dataset.permute()

        # nothing in the code for each batch is now any different than base PyTorch
        for i, (batch_x, batch_y) in enumerate(loader):
            optimizer.zero_grad()
            batch_y_pred = model(batch_x)

            loss = criterion(batch_y_pred.transpose(1, 2), batch_y)
            loss.backward()
            optimizer.step()

            logger.info(
                f"Worker {worker_rank} - {datetime.datetime.now().isoformat()} - epoch {epoch} - batch {i} - batch complete - loss {loss.item()}"
            )

        # the first rh call saves a json file with the loss from the worker at the end of the epoch
        rh.submit_result(
            f"logs/data_{worker_rank}_{epoch}.json",
            json.dumps(
                {
                    "loss": loss.item(),
                    "time": datetime.datetime.now().isoformat(),
                    "epoch": epoch,
                    "worker": worker_rank,
                }
            ),
        )
        # this saves the model. We only need to do it for one worker (so we picked worker 0)
        if worker_rank == 0:
            rh.submit_result("model.pkl", pickle.dumps(model.state_dict()))

n_workers = 3
cluster = SaturnCluster(n_workers=n_workers)
client = Client(cluster)
client.wait_for_workers(n_workers)

key = uuid.uuid4().hex
rh = results.DaskResultsHandler(key)

futures = dispatch.run(client, train)
rh.process_results("/home/radia/MiniBart/sample_results", futures, raise_errors=False)

client.close()