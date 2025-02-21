import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os
from paule.paule import Paule
from paule.models import ForwardModel
from tqdm import tqdm
import pandas as pd
import pickle
import logging

logging.basicConfig(level=logging.INFO)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class RMSELoss(torch.nn.Module):
    """
    Root-Mean-Squared-Error-Loss (RMSE-Loss) Taken from Paul, it is  from stackoverflow
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class ForwardDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        normalized_melspec = row["melspec_norm_recording"]
        cp_normalized = row["cp_norm"]
        return cp_normalized, normalized_melspec


def train_forward_on_one_df(
    batch_size=8,
    lr=1e-3,
    device="cuda",
    file_path="",
    criterion=None,
    optimizer_module=None,
    forward_model=None,
):

    df_train = pd.read_pickle(file_path)
    dataset = ForwardDataset(df_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    forward_model.train()
    pytorch_total_params = sum(
        p.numel() for p in forward_model.parameters() if p.requires_grad
    )
    logging.info("Trainable Parameters in Model: %s", pytorch_total_params)

    criterion = RMSELoss()
    optimizer = optimizer_module(forward_model.parameters(), lr=lr)

    for batch in tqdm(dataloader):
        cp, melspec = batch
        cp = cp.to(device)
        melspec = melspec.to(device)

        optimizer.zero_grad()
        output = forward_model(cp)
        loss = criterion(output, melspec)
        loss.backward()
        optimizer.step()


def train_whole_dataset(
    batch, lr, device, criterion, optimizer_module, data_path, epochs=10
):
    files = os.listdir(data_path)
    filtered_files = [file for file in files if file.endswith(".pkl")]

    forward_model = (
        ForwardModel(
            num_lstm_layers=1,
            hidden_size=720,
            input_size=30,
            output_size=60,
            apply_half_sequence=True,
        )
        .double()
        .to(DEVICE)
    )
    for epoch in range(epochs):
        shuffeled_files = np.random.shuffle(filtered_files)
        for file in shuffeled_files:
            train_forward_on_one_df(
                batch_size=batch,
                lr=lr,
                device=device,
                file_path=os.path.join(data_path, file),
                criterion=criterion,
                optimizer_module=optimizer_module,
                forward_model=forward_model,
            )
