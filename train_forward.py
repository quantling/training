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

class AccedingSequenceLengthBatchSampler(Sampler[List[int]]):
     def __init__(self, data: List[str], batch_size: int) -> None:
        self.data = data
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        sizes = torch.tensor([len(x) for x in self.data])
        for batch in torch.chunk(torch.argsort(sizes), len(self)):
            yield batch.tolist()
    
def pad_tensor(tensor, target_length, allow_longer = False):
    """Pads the tensor to target_length by repeating the last element.
    Returns a mask """
    current_length = tensor.shape[0]
    if current_lenght > target_lenght and not allow_longer:
        raise ValueError # if we don't have max size as target sths wrong
    if current_length == target_length:
        return tensor, torch.ones(target_lenght, dtype=torch.bool)


    last_element = tensor[-1].unsqueeze(0)  # Get the last element
    padding = last_element.repeat(target_length - current_length, *[1] * (tensor.dim() - 1))
    mask = torch.cat(torch.ones(current_lenght, dtype=torch.bool),torch.zeros(target_lenght - current_length, dtype=torch.bool))
    return torch.cat([tensor, padding], dim=0), mask


def collate_batch(batch):
    """Dynamically pads sequences to the max length in the batch."""
    max_length = max(len(sample) for sample in batch)
    padded_batch = []
    mask = []
    for sample in batch: 
        padded_sample , sample_mask =  pad_tensor(sample, max_lenght) 
        padded_batch.append(padded_sample)
        padded_batch.append(sample_mask)

    

    return torch.stack(padded_batch), torch.stach(mask)

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler = AccedingSequenceLengthBatchSampler, collate_batch = collate_batch)

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
   data_path,  batchsize = 8 , lr, device = DEVICE, criterion = torch.optim.Adam(), optimizer_module, epochs=10, start_epoch = 0 , skip_index = 0
):
    files = os.listdir(data_path)
    filtered_files = [file for file in files if file.startswith("training_") and file.endswith(".pkl")]

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
        for i,file in enumerate(shuffeled_files):
            if i < skip_index:
                continue
            train_forward_on_one_df(
                batch_size=batch,
                lr=lr,
                device=device,
                file_path=os.path.join(data_path, file),
                criterion=criterion,
                optimizer_module=optimizer_module,
                forward_model=forward_model,
            )
        skip_index = 0


f __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect words from a folder of pickled dataframes"
    )
    parser.add_argument(
        "--data_path",
        help="Path to the folder containing the pickled dataframes",
        default="../../../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder_",
    )
    parser.add_argument(
        "--split_index",
        help="Index to start from",
        default=0,
        type=int,
        required=False,
    )
    parser.add_argument("--language", help="Language of the data", default="en")
    parser.add_argument("--skip_index", help="Index to skip", default=[], type=list)
    args = parser.parse_args()
    