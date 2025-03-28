import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os
import logging
import pandas as pd
import pickle
import argparse
import math
from tqdm import tqdm
from training_utils import RMSELoss, AccedingSequenceLengthBatchSampler, pad_tensor, validate_whole_dataset, plot_validation_losses
from paule.models import InverseModelMelTimeSmoothResidual as InverseModel
import gc

logging.basicConfig(level=logging.INFO)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Inverse Mode : Melspec -> CP
# In this file I have kept the order of cps and melspecs as in the code for the forward model to avoid confusion for myself
class InverseModelDataset(Dataset):
    """
    Dataset for the inverse model, which takes melspecs and outputs CPs. Essentially the same as the forward model dataset, but sizes are different
    since take Melspecs as input and CPs as output."""
    def __init__(self, df):
        self.df = df

        #convert the melspecs to tensors
        self.df["melspec_norm_synthesized"] = self.df["melspec_norm_synthesized"].apply(
            lambda x: torch.tensor(x, dtype=torch.float64)
        )
        self.melspecs = self.df["melspec_norm_synthesized"].tolist()
        self.df["cp_norm"] = self.df["cp_norm"].apply(
            lambda x: torch.tensor(x, dtype=torch.float64)
        )
        self.cp_norm = self.df["cp_norm"].tolist()
        self.sizes = [len(x) for x in self.melspecs]  

    def __len__(self):
        logging.debug(f"len of df: {len(self.df)}")
        return len(self.df)

    def __getitem__(self, idx):
        logging.debug(f"idx: {idx}")
       
      
        logging.debug("successfully converted melspec to tensor")
        return self.cp_norm[idx], self.melspecs[idx]

def collate_batch_with_padding_inverse_model(batch):
    """
    Dynamically pads sequences to the max length in the batch for the inverse model.
    
    # TODO: Customize padding strategy based on your specific input/target requirements
    """
    # Determine max lengths
 
    max_length_cps = max( len(sample[0]) for sample in batch)
    max_length_melspecs =math.ceil(max_length_cps /2) 
    
    padded_cps = []
    padded_melspecs = []
    cp_masks = []
    melspec_masks = []
    last_cps_indices = []
    last_melspecs_indices = []
    
    for sample in batch:
       
        padded_cp, cp_mask = pad_tensor(sample[0], max_length_cps)
        padded_melspec, melspec_mask = pad_tensor(sample[1], max_length_melspecs)
        logging.debug(f"padded_cp: {padded_cp} vs original: {sample[0]}")
        logging.debug(f"padded_melspec: {padded_melspec} vs original: {sample[1]}")
        last_cp_indice = cp_mask.long().argmax(dim=-1)
        last_melspec_indice = melspec_mask.long().argmax(dim=-1)

        logging.debug(f"last indices: {last_cp_indice} vs {last_melspec_indice}")
        
        last_cps_indices.append(last_cp_indice)
        last_melspecs_indices.append(last_melspec_indice)
        
        padded_cps.append(padded_cp)
        padded_melspecs.append(padded_melspec)
        cp_masks.append(cp_mask)
        melspec_masks.append(melspec_mask)
    
    return (
        torch.stack(padded_cps), 
        torch.stack(padded_melspecs), 
        torch.stack(cp_masks), 
        torch.stack(melspec_masks),
        torch.stack(last_cps_indices),
        torch.stack(last_melspecs_indices)
    )

def train_inverse_model_on_one_df(
    batch_size=8,
    lr=1e-3,
    device="cuda",
    file_path="",
    criterion=None,
    optimizer=None,
    inverse_model=None,
):
    """
    Train the inverse model on a single dataframe
    
    # TODO: Customize training logic as needed
    """
    df_train = pd.read_pickle(file_path)
    logging.info(f"Creating dataset from {file_path}")
    dataset = InverseModelDataset(df_train)
    
    # TODO: Potentially customize sampling strategy
    sampler = AccedingSequenceLengthBatchSampler(dataset, batch_size)
    
    dataloader = DataLoader(
        dataset, 
        batch_sampler=sampler,
        collate_fn=collate_batch_with_padding_inverse_model
    )

    inverse_model.train()
    pytorch_total_params = sum(
        p.numel() for p in inverse_model.parameters() if p.requires_grad
    )
    logging.info("Trainable Parameters in Model: %s", pytorch_total_params)

    if optimizer is None:
        raise ValueError("Optimizer is None")
    if criterion is None:
        raise ValueError("Criterion is None")

    for batch in tqdm(iter(dataloader)):
        optimizer.zero_grad()
        
        # TODO: Unpack batch according to your specific needs
        cps, melspecs , _, _, last_input_indices, _ = batch
        
        cps = cps.to(device)
        melspecs = melspecs.to(device)
        # Model forward pass
        logging.debug(f"input: {melspecs.shape} (melspec) ")
        logging.debug(f"target {cps.shape} (cps) ")
       
        output = inverse_model(melspecs)
        if output.shape[1] != cps.shape[1]:
            logging.debug(f"Shapes are output :{output.shape} and  melspec: {melspecs.shape} and cp shape is {cps.shape}")
            if output.shape[1] > cps.shape[1]:
                
                output = output[:, :cps.shape[1], :]
                logging.debug("Had to cut the output")
             
           
        logging.debug(f"output: {output.shape} vs target( cps): {cps.shape}")
      
        # Compute loss
        loss = criterion(output, cps)
        loss.backward()
        optimizer.step()
        
        logging.debug(f"loss: {loss.item()}")

def train_inverse_model_on_whole_dataset(
    data_path,  
    batch_size=8, 
    lr=1e-4, 
    device=DEVICE, 
    criterion=None, 
    optimizer_module=None, 
    epochs=10, 
    start_epoch=0, 
    skip_index=0, 
    validate_every=1, 
    save_every=1,
    language="",
    load_from="",
    **kwargs  # Additional arguments for flexibility
):
    """
    Train the inverse model across multiple dataframes
    
   
    """
    if criterion is None:
      
        criterion =RMSELoss()
    
    if optimizer_module is None:
        optimizer_module = optim.Adam
    
    
    inverse_model = InverseModel(
       num_lstm_layers=1,
        input_size=60,
        output_size=30,
        hidden_size=720,
        
    ).double()
    
    optimizer = optimizer_module(inverse_model.parameters(), lr=lr)
    
    if load_from:
        inverse_model.load_state_dict(torch.load(load_from))
        optimizer.load_state_dict(torch.load(f"optimizer_{language}.pt"))

    inverse_model.to(device)
    
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
            train_inverse_model_on_one_df(
                batch_size=batch_size,
                lr=lr,
                device=device,
                file_path=os.path.join(data_path, file),
                criterion=criterion,
                optimizer=optimizer,
                inverse_model=inverse_model,
            )
        
        if epoch % validate_every == 0:
            logging.info(f"Validating on {len(validation_files)} files")
            # TODO: Implement validate_whole_dataset or validation logic
            mean_loss, std_loss = validate_whole_dataset(
                validation_files,
                data_path,
                batch_size=batch_size,
                device=device,
                criterion=criterion,
                model=inverse_model,
                validate_on_one_df=validate_inverse_model_on_one_df,
                model_name="inverse_model",
            )
            logging.info(f"Mean validation loss: {mean_loss}, Std loss: {std_loss}")
            validation_losses.append(mean_loss)
            
           
            plot_validation_losses(validation_losses, language, model_name="inverse_model")
        
        if epoch % save_every == 0 or epoch == epochs - 1:
            torch.save(inverse_model.state_dict(), f"inverse_model_{language}.pt")
            torch.save(optimizer.state_dict(), f"optimizer_inverse_model_{language}.pt")

    logging.info("Finished training the inverse model")

def validate_inverse_model_on_one_df(
    batch_size=8,
    lr=1e-3,
    device="cuda",
    file_path="",
    criterion=None,
    model=None,
):
    """
    Validate the inverse model on a single dataframe
    
    # TODO: Customize validation logic
    """
    if criterion is None:
       raise ValueError("Criterion is None")
    
    if model is None:
        raise ValueError("Model is None")
    
    df_validate = pd.read_pickle(file_path)
    logging.info(f"Creating dataset from {file_path}")
    
    dataset = InverseModelDataset(df_validate)
    sampler = AccedingSequenceLengthBatchSampler(dataset, batch_size)
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_batch_with_padding_inverse_model
    )

    model.eval()
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logging.info("Trainable Parameters in Model: %s", pytorch_total_params)
    
    losses = []
    
    with torch.no_grad():
        for batch in tqdm(iter(dataloader)):
             
       
        
            # TODO: Unpack batch according to your specific needs
            cps, melspecs , _, _, last_input_indices, _ = batch
            cps = cps.to(device)
            melspecs = melspecs.to(device)
            # Model forward pass
            logging.debug(f"input: {melspecs.shape} (melspec) ")
            logging.debug(f"target {cps.shape} (cps) ")
           
           
        
            # Compute loss
           
          
            
            output = model(melspecs)
            logging.debug(f"output: {output.shape} vs target( cps): {cps.shape}")
            if output.shape[1] != cps.shape[1]:
                logging.debug(f"Shapes are output :{output.shape} and  melspec: {melspecs.shape} and cp shape is {cps.shape}")
                if output.shape[1] > cps.shape[1]:
                    
                    output = output[:, :cps.shape[1], :]
                    logging.debug("Had to cut the output")
            loss = criterion(output, cps)
            
            losses.append(loss.item())
            logging.debug(f"loss: {loss.item()}")
    
    return np.mean(losses), np.std(losses), losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # TODO: Update argument descriptions and defaults as needed
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="The learning rate for the training",
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        help="The path to the data",
        default="../../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder_"
    )
    parser.add_argument(
        "--language",
        type=str,
        help="The language of the data",
        default="de",
    )
    parser.add_argument("--debug", action="store_true", help="Set the logging level to debug")
    parser.add_argument("--seed", type=int, help="The seed for the random number generator", default=42)
    parser.add_argument("--testmode", help="Test mode", action="store_true")
    parser.add_argument("--optimizer", help="Optimizer to use", default="adam")
    parser.add_argument("--criterion", help="Criterion to use", default="mse")
    parser.add_argument("--batch_size", help="Batch size", default=8, type=int)
    parser.add_argument("--epochs", help="Number of epochs", default=10, type=int)
    parser.add_argument("--validate_every", help="Validate every n epochs", default=1, type=int)
    parser.add_argument("--save_every", help="Save every n epochs", default=1, type=int)

    args = parser.parse_args()

    # TODO: Customize optimizer and criterion selection
    if args.optimizer == "adam":
        optimizer_module = optim.Adam
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    if args.criterion == "mse":
        criterion = nn.MSELoss()
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

    # TODO: Add any specific configuration or preprocessing
    # minimum_distance = pickle.load(open(f"min_distance_{args.language}.pkl", "rb"))
    
    data_path = args.data_path + args.language
    
    train_inverse_model_on_whole_dataset(
        data_path=data_path,
        batch_size=args.batch_size,
        lr=args.lr,
        language=args.language,
        criterion=criterion,
        optimizer_module=optimizer_module,
        epochs=args.epochs,
        validate_every=args.validate_every,
        save_every=args.save_every,
    )
    
    logging.info("Finished training the inverse model")