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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
      
        row = self.df.iloc[idx]
        fast_text_vector = row["vector"]  
        log_mel_spektrogram = row["melspec_norm_recording"]  
       
        
       
        return fast_text_vector, log_mel_spektrogram
    

def train_embedder_on_one_df( batch_size=32, epochs=10, lr=1e-3, device='cuda'):
    pass