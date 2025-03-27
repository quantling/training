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
from training_utils import RMSELoss, AccedingSequenceLengthBatchSampler, pad_tensor, validate_whole_dataset, plot_validation_losses
import logging
import pandas as pd
import pickle
import torch
import argparse



logging.basicConfig(level=logging.INFO)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
        self.sizes =  [len(x) for x in self.melspecs]
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
      
        return self.melspecs[idx],self.vectors[idx]
       
        
       
    
def collate_batch_with_padding_embedder(batch):
    """Dynamically pads sequences to the max length in the batch. It is specifically for the embedder model."""
    logging.debug(f"batch in collate_batch: {batch}")
    max_length_melspecs = max(len(sample[0]) for sample in batch)
    max_length_vector = max( len(sample[1]) for sample in batch)
  
    logging.debug(f"max_length_melspecs: {max_length_melspecs}")
    logging.debug(f"max_length_vectors: {max_length_vector}")
    padded_melspecs= []
    sample_vectors= []
    mask = []
    last_indices = []
    for sample in batch: 
       
        padded_melspec , sample_mask =  pad_tensor(sample[0], max_length_melspecs) 
        sample_vector = sample[1]
        logging.debug(f"mask: {sample_mask}")
       
        last_indice = sample_mask.long().argmax(dim=-1)
        last_indices.append(last_indice) 
        sample_vectors.append(sample_vector)
        padded_melspecs.append(padded_melspec)
        mask.append(sample_mask)

    

    return torch.stack(padded_melspecs), torch.stack(sample_vectors), torch.stack(mask), torch.stack(last_indices)

def train_embedder_on_one_df(
    batch_size=8,
    lr=1e-3,
    device="cuda",
    file_path="",
    criterion=None,
    optimizer=None,
    embedding_model=None,
):

    df_train = pd.read_pickle(file_path)
    logging.info(f"Creating dataset from {file_path}")
    dataset = EmbedderDataset(df_train)
    sampler = AccedingSequenceLengthBatchSampler(dataset, batch_size)
    logging.info(f"Creating dataloader from {file_path}")
    dataloader = DataLoader(
    dataset, 
    batch_sampler=sampler,  # Use batch_sampler instead of batch_size and sampler
    collate_fn=collate_batch_with_padding_embedder)

    embedding_model.train()
    pytorch_total_params = sum(
        p.numel() for p in embedding_model.parameters() if p.requires_grad
    )
    logging.info("Trainable Parameters in Model: %s", pytorch_total_params)

    #is the optimizer updated from the last df?
    if optimizer is None:
        raise ValueError("Optimizer is None")
    if criterion is None:
        raise ValueError("Criterion is None")

    for batch in tqdm(iter(dataloader)):
        #logging.debug(batch)
        optimizer.zero_grad()
        melspecs, vectors,_, last_indices = batch
        melspecs = melspecs.to(device)
        vectors = vectors.to(device)
        random_added = torch.tensor(np.random.normal(0, minimum_distance, vectors.shape)).to(device)
        logging.debug(f"Random vector {random_added}")
        vectors = vectors + random_added
        logging.debug(f"vectors: {vectors.shape}")
        logging.debug(f"melspecs: {melspecs.shape}")
        output = embedding_model(melspecs, last_indices)
        logging.debug(f"output: {output.shape}")
        loss = criterion(output, vectors)
        loss.backward()
        optimizer.step()
        logging.debug(f"loss: {loss.item()}")
    

def train_embedder_on_whole_dataset(
         data_path,  batch_size = 8 , lr = 1e-4, device = DEVICE, criterion = None, optimizer_module= None, epochs=10, start_epoch = 0 , skip_index = 0, validate_every = 1, save_every = 1 ,language = "",
   load_from = "", minimum_distance = None
):
    if criterion is None:
        criterion = RMSELoss()
    if optimizer_module is None:
        optimizer_module = optim.Adam
    if minimum_distance is None:
        raise ValueError("minimum_distance is None")
    
    embedding_model = EmbeddingModel(input_size=60,
                            num_lstm_layers=2,
                            hidden_size=360,
                            dropout=0.7,
                            post_upsampling_size=0).double() # right now just taken from Pauls old code
    optimizer = optimizer_module(embedding_model.parameters(), lr=lr)
    if load_from:	
        embedding_model.load_state_dict(torch.load(load_from))
        optimizer.load_state_dict(torch.load(f"optimizer_{language}.pt"))

    embedding_model.to(device)
    sorted_files = sorted(os.listdir(data_path))
    validation_files = [file for file in sorted_files if file.startswith("validation_") and file.endswith(".pkl")]

    filtered_files = [file for file in sorted_files if file.endswith(".pkl") and "train" in file]
    validation_losses = []
    for epoch in tqdm(range(start_epoch, epochs)):
        logging.info(f"Epoch {epoch}")
        np.random.shuffle(filtered_files)
        for i, file in enumerate(filtered_files):
            logging.info(f"Processing {i}th file out of {len(filtered_files)}")
            if i < skip_index:
                continue
            logging.info(f"Training on {file}")
            train_embedder_on_one_df(
                batch_size=batch_size,
                lr=lr,
                device=device,
                file_path=os.path.join(data_path, file),
                criterion=criterion,
                optimizer=optimizer,
                embedding_model=embedding_model,
            )
            logging.info(f"Finished training on {file}")
        if epoch % validate_every == 0:
                logging.info(f"Validating on {len(validation_files)} files")
                logging.debug(f"Model type: {embedding_model}")
                logging.debug(f"Criteron type: {criterion}")
                mean_loss, std_loss = validate_whole_dataset(validation_files,data_path,
                batch_size=batch_size,
                device=device,
                criterion=criterion,
                model=embedding_model,
                validate_on_one_df=validate_embedder_on_one_df,)
                logging.info(f"Mean valdiation loss: {mean_loss}, Std loss: {std_loss}")
                validation_losses.append(mean_loss)
                plot_validation_losses(validation_losses, language, model_name="embedder")
        if epoch % save_every == 0 or epoch == epochs - 1:
                torch.save(embedding_model.state_dict(), f"embedding_model_{language}.pt")
                torch.save(optimizer.state_dict(), f"optimizer_{language}.pt")

    logging.info("Finished training the embedder")



def validate_embedder_on_one_df(
    batch_size=8,
    lr=1e-3,
    device="cuda",
    file_path="",
    criterion=None,
    model=None,
):
    if criterion is None:
       raise ValueError("Criterion is None")
    
    if model is None:
        raise ValueError("Model is None")
    df_validate = pd.read_pickle(file_path)
    logging.info(f"Creating dataset from {file_path}")
    dataset = EmbedderDataset(df_validate)
    sampler = AccedingSequenceLengthBatchSampler(dataset, batch_size)
    logging.info(f"Creating dataloader from {file_path}")
    
    dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,  # Use batch_sampler instead of batch_size and sampler
    collate_fn=collate_batch_with_padding_embedder) 

    model.eval()
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logging.info("Trainable Parameters in Model: %s", pytorch_total_params)
    losses = []
    for batch in tqdm(iter(dataloader)):
        #logging.debug(batch)
      
        melspecs, vectors,_, last_indices = batch
        melspecs = melspecs.to(device)
        vectors = vectors.to(device)
        np.random.normal(0, minimum_distance, vectors.shape)
        logging.debug(f"vectors: {vectors.shape}")
        logging.debug(f"melspecs: {melspecs.shape}")
        output = model(melspecs, last_indices)
        logging.debug(f"output: {output.shape}")
        loss = criterion(output, vectors)
        losses.append(loss.item())
        logging.debug(f"loss: {loss.item()}")
    return np.mean(losses), np.std(losses),losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="The learning rate for the training",
    )
    parser.add_argument("--data_path", type=str, help="The path to the data",default="../../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder_" )
    parser.add_argument(
        "--language",
        type=str,
        help="The language of the word vectors",
        default="de",
    )
    parser.add_argument("--debug", action="store_true", help="Set the logging level to debug")
    parser.add_argument("--seed", type=int, help="The seed for the random number generator", default=42)
    parser.add_argument("--testmode", help="Test mode", action="store_true")
    parser.add_argument("--optimizer", help="Optimizer to use", default="adam")
    parser.add_argument("--criterion", help="Criterion to use", default="rmse")
    parser.add_argument("--batch_size", help="Batch size", default=8, type=int)
    parser.add_argument("--epochs", help="Number of epochs", default=10, type=int)
    parser.add_argument("--validate_every", help="Validate every n epochs", default=1, type=int)
    parser.add_argument("--save_every", help="Save every n epochs", default=1, type=int)


    args = parser.parse_args()
    if args.optimizer == "adam":
        optimizer_module = optim.Adam
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    if args.criterion == "rmse":
        criterion = RMSELoss()
    else:
        raise ValueError(f"Criterion {args.criterion} not supported")
    if args.testmode:
        data_path = "../../../../../../mnt/Restricted/Corpora/CommonVoiceVTL/mini_corpus_" 
        args.data_path = data_path
        logging.info(f"Test mode: {args.data_path}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    minimum_distance = pickle.load(open(f"min_distance_{args.language}.pkl", "rb"))
    data_path = args.data_path + args.language
    train_embedder_on_whole_dataset(
        data_path=data_path,
        batch_size=args.batch_size,
        lr=args.lr,
        language=args.language,
        minimum_distance=minimum_distance,
        criterion=criterion,
        optimizer_module=optimizer_module,
        epochs=args.epochs,
        validate_every=args.validate_every,
        save_every=args.save_every,
    )
    logging.info("Finished training the embedder")
    logging.info(f"The minimum distance between vectors was {minimum_distance}")