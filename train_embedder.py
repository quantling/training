import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os
from paule.paule import Paule
from paule.models import EmbeddingModel
from tqdm import tqdm
from train_forward import RMSELoss, AccedingSequenceLengthBatchSampler, pad_tensor
import logging
import pandas as pd
import pickle
import torch

class EmbedderDataset(Dataset):
    def __init__(self, df):
        """
        Args:
            df (pd.DataFrame): DataFrame with at least the following columns:
                - 'waveform': a NumPy array representing the audio waveform.
                - 'sampling_rate': the audio's sampling rate.
                - 'length': (optional) precomputed length of the waveform.
        """
        self.df = df
        self.df["melspec_norm_recorded"] = self.df["melspec_norm_recorded"].apply(
            lambda x: torch.tensor(x, dtype=torch.float64)
        )
        self.melspecs = self.df["melspec_norm_recorded"].tolist()
        self.df["vector"] = self.df["vector"].apply(
            lambda x: torch.tensor(x, dtype=torch.float64)
        )
        self.vectors = self.df["vector"].tolist()
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
      
        return self.melspecs[idx],self.vectors[idx]
       
        
       
    
def collate_batch_with_padding_embedder(batch):
    """Dynamically pads sequences to the max length in the batch. It is specifically for the embedder model."""
    logging.debug(f"batch in collate_batch: {batch}")
    max_length_melspecs = max(len(sample[0]) for sample in batch)
    max_length_vector = max( len(sample[1]) for sample in batch)
    max_length = max(max_length_melspecs, max_length_vector)
    logging.debug(f"max_length: {max_length}")
    
    logging.debug(f"max_length_melspecs: {max_length_melspecs}")
    logging.debug(f"max_length_vectors: {max_length_vector}")
    padded_cps= []
    sample_vectors= []
    mask = []
    for sample in batch: 
       
        padded_melspec , sample_mask =  pad_tensor(sample[0], max_length) 
        sample_vector = sample[1]
        sample_vectors.append(padded_melspec)
        padded_cps.append(sample_vector)
        mask.append(sample_mask)

    

    return torch.stack(padded_cps), torch.stack(sample_vectors)

def train_embedder_on_one_df(
    batch_size=8,
    lr=1e-3,
    device="cuda",
    file_path="",
    criterion=None,
    optimizer=None,
    embedder_model=None,
):

    df_train = pd.read_pickle(file_path)
    dataset = EmbedderDataset(df_train)
    sampler = AccedingSequenceLengthBatchSampler(dataset, batch_size)
    dataloader = DataLoader(
    dataset, 
    batch_sampler=sampler,  # Use batch_sampler instead of batch_size and sampler
    collate_fn=collate_batch_with_padding_embedder
)

    embedder_model.train()
    pytorch_total_params = sum(
        p.numel() for p in embedder_model.parameters() if p.requires_grad
    )
    logging.info("Trainable Parameters in Model: %s", pytorch_total_params)

    #is the optimizer updated from the last df?
    if optimizer is None:
        raise ValueError("Optimizer is None")
    if criterion is None:
        raise ValueError("Criterion is None")

    for batch in iter(dataloader):
        #logging.debug(batch)
        optimizer.zero_grad()
        melspecs, vectors = batch
        melspecs = melspecs.to(device)
        vectors = vectors.to(device)

