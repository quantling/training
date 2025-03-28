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
import gc
import argparse
import math
from training_utils import RMSELoss, AccedingSequenceLengthBatchSampler, pad_tensor, validate_whole_dataset, plot_validation_losses

logging.basicConfig(level=logging.INFO)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ForwardDataset(Dataset):
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
        self.sizes = [len(x) for x in self.cp_norm]  # Assuming cp_norm has the sequence lengths

    def __len__(self):
        logging.debug(f"len of df: {len(self.df)}")
        return len(self.df)

    def __getitem__(self, idx):
        logging.debug(f"idx: {idx}")
       
      
        logging.debug("successfully converted melspec to tensor")
        return self.cp_norm[idx], self.melspecs[idx]
       

def collate_batch_with_padding(batch):
    """Dynamically pads sequences to the max length in the batch. It is specifically for the forward model."""
    logging.debug(f"batch in collate_batch: {batch}")
    
    max_length_cps = max( len(sample[0]) for sample in batch)
    max_length_melspecs =math.ceil(max_length_cps /2)
    logging.debug(f"max_length_melspecs: {max_length_melspecs}")
    logging.debug(f"max_length_cps: {max_length_cps}")
    padded_cps= []
    padded_melspecs = []
    mask = []
    for sample in batch: 
        logging.debug(f"sample shape: {sample[0].shape}")
        padded_melspec , sample_mask =  pad_tensor(sample[1], max_length_melspecs) 
        padded_cp, _ = pad_tensor(sample[0], max_length_cps)
        logging.debug(f"padded_melspec shape: {padded_melspec.shape}")
        logging.debug(f"padded_cp shape: {padded_cp.shape}")
        #assert padded_cp.shape[0] == padded_melspec.shape[0] * 2 if padded_cp.shape[0] % 2 == 0 else  padded_cp.shape[0] == (padded_melspec.shape[0] * 2) -1 , f"Shapes are  cp : {padded_cp.shape} and  melspec: {padded_melspec.shape}"
        padded_melspecs.append(padded_melspec)
        padded_cps.append(padded_cp)
        mask.append(sample_mask)

    

    return torch.stack(padded_cps), torch.stack(padded_melspecs)


def train_forward_on_one_df(
    batch_size=8,
    lr=1e-3,
    device="cuda",
    file_path="",
    criterion=None,
    optimizer=None,
    forward_model=None,
):

    df_train = pd.read_pickle(file_path)
    dataset = ForwardDataset(df_train)
    sampler = AccedingSequenceLengthBatchSampler(dataset, batch_size)
    dataloader = DataLoader(
    dataset, 
    batch_sampler=sampler,  # Use batch_sampler instead of batch_size and sampler
    collate_fn=collate_batch_with_padding
)

    forward_model.train()
    pytorch_total_params = sum(
        p.numel() for p in forward_model.parameters() if p.requires_grad
    )
    logging.info("Trainable Parameters in Model: %s", pytorch_total_params)
    
    #is the optimizer updated from the last df?
    if optimizer is None:
        raise ValueError("Optimizer is None")
    if criterion is None:
        raise ValueError("Criterion is None")

    for batch in tqdm(iter(dataloader)):
        #logging.debug(batch)
        
        cp, melspec = batch
        cp = cp.to(device)
        melspec = melspec.to(device)
    
        assert cp.shape[2] == 30 and melspec.shape[2] == 60, f"Shapes are {cp.shape} and {melspec.shape}"
        optimizer.zero_grad()
        output = forward_model(cp)
        logging.debug(f"output shape: {output.shape}")
        logging.debug(f"cp shape: {cp.shape}")
        logging.debug(f"melspec shape: {melspec.shape}")
        if output.shape[1] != melspec.shape[1]:
            logging.debug(f"Shapes are output :{output.shape} and  melspec: {melspec.shape} and cp shape is {cp.shape}")
            melspec = melspec[:, :output.shape[1], :] #but what happens here?
        assert output.shape[1] == melspec.shape[1], f"Shapes are output :{output.shape} and  melspec: {melspec.shape} and cp shape is {cp.shape}"
        loss = criterion(output, melspec)
        loss.backward()
        optimizer.step()
        logging.debug(f"loss: {loss.item()}")


def validate_forward_on_one_df(
    batch_size=8,
    device="cuda",
    file_path="",
    criterion=None,
    model=None,
):

    df_train = pd.read_pickle(file_path)
    dataset = ForwardDataset(df_train)
    sampler = AccedingSequenceLengthBatchSampler(dataset, batch_size)
    dataloader = DataLoader(
            dataset, 
            batch_sampler=sampler,  # Use batch_sampler instead of batch_size and sampler
            collate_fn=collate_batch_with_padding
    )

    model.eval()
    
    losses = []
   
    
    if criterion is None:
        raise ValueError("Criterion is None")

    for batch in iter(dataloader):
        #logging.debug(batch)
        cp, melspec = batch
        cp = cp.to(device)
        melspec = melspec.to(device)
        cp = cp.squeeze(1)
        melspec = melspec.squeeze(1)
        assert cp.shape[2] == 30 and melspec.shape[2] == 60, f"Shapes are {cp.shape} and {melspec.shape}"
        output = model(cp)
        logging.debug(f"cp shape: {cp.shape}")
        logging.debug(f"output shape: {output.shape}")
        logging.debug(f"melspec shape: {melspec.shape}")
        assert output.shape[1] == melspec.shape[1], f"Shapes are {output.shape} and {melspec.shape}"
        loss = criterion( melspec,output)

     
        losses.append(loss.item())
    

    return np.mean(losses), np.std(losses), losses



def train_whole_dataset(
   data_path,  batch_size = 8 , lr = 1e-4, device = DEVICE, criterion = None, optimizer_module= None, epochs=10, start_epoch = 0 , skip_index = 0, validate_every = 1, save_every = 1 ,language = "",
   load_from = ""
):  
    data_path = data_path + args.language
    files = os.listdir(data_path)
    #print(files)
    filtered_files = [file for file in files if file.startswith("training_") and file.endswith(".pkl")]
    validation_files = [file for file in files if file.startswith("validation_") and file.endswith(".pkl")]
    print(filtered_files)
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
    optimizer = optimizer_module(forward_model.parameters(), lr=lr)
    if load_from != "":
        forward_model.load_state_dict(torch.load(load_from))
        optimizer.load_state_dict(torch.load(load_from.replace("forward_model", "optimizer")))
        validation_losses = pickle.load(open(load_from.replace("forward_model", "validation_losses"), "rb"))
        epoch = pickle.load(open(load_from.replace("forward_model", "epoch"), "rb"))
        skip_index = pickle.load(open(load_from.replace("forward_model", "skip_index"), "rb"))
        logging.info(f"Loaded model from {load_from}")
    validation_losses = []
    for epoch in tqdm(range(epochs)):
        np.random.shuffle(filtered_files)
        shuffeled_files = filtered_files
        print(shuffeled_files)
        for i,file in enumerate(shuffeled_files):
            logging.info(f"Processing {i}th file out of {len(shuffeled_files)}")
            if i < skip_index:
                continue
            logging.info(f"Training on {file}")
            train_forward_on_one_df(
                batch_size=batch_size,
                lr=lr,
                device=device,
                file_path=os.path.join(data_path, file),
                criterion=criterion,
                optimizer=optimizer,
                forward_model=forward_model,
            )
            gc.collect()

        if epoch % validate_every == 0:
            mean_loss, std_loss = validate_whole_dataset(
                validation_files,
                data_path,
                batch_size=batch_size,
                device=device,
                criterion=criterion,
                optimizer_module=optimizer_module,
                model=forward_model,
                validate_on_one_df=validate_forward_on_one_df,
            )
            logging.info(f"Mean valdiation loss: {mean_loss}, Std loss: {std_loss}")
            validation_losses.append(mean_loss)
            plot_validation_losses(validation_losses, language, save_path=".")
        if epoch % save_every == 0 or epoch == epochs - 1:
            model_name = f"forward_model_{args.language}_{epoch}.pt"
            os.makedirs("models", exist_ok=True)
            model_dir = os.path.join("models", model_name)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(forward_model.state_dict(), os.path.join(model_dir,f"forward_model_{language}_{epoch}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(model_dir, f"optimizer_{language}_{epoch}.pt"))
            pickle.dump(validation_losses, open(os.path.join(model_dir,f"validation_losses_{language}.pkl"), "wb"))
            pickle.dump(epoch, open(os.path.join(model_dir,f"epoch_{language}.pkl"), "wb"))
            pickle.dump(skip_index, open(os.path.join(model_dir,f"skip_index_{language}.pkl"), "wb"))
            logging.info(f"Saved model, validation losses and epoch")
        skip_index = 0

    logging.info("Finished training")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect words from a folder of pickled dataframes"
    )
    parser.add_argument(
        "--data_path",
        help="Path to the folder containing the pickled dataframes",
        default="../../../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder_",
    )
    parser.add_argument(
        "--skip_index",
        help="Index to start from",
        default=0,
        type=int,
        required=False,
    )
    parser.add_argument("--start_epoch", help="Epoch to start from", default=0, type=int)
    parser.add_argument("--language", help="Language of the data", default="de")

    parser.add_argument("--optimizer", help="Optimizer to use", default="adam")
    parser.add_argument("--criterion", help="Criterion to use", default="rmse")
    parser.add_argument("--batch_size", help="Batch size", default=8, type=int)
    parser.add_argument("--lr", help="Learning rate", default=1e-4, type=float)
    parser.add_argument("--epochs", help="Number of epochs", default=10, type=int)
    parser.add_argument("--validate_every", help="Validate every n epochs", default=1, type=int)
    parser.add_argument("--save_every", help="Save every n epochs", default=1, type=int)
    parser.add_argument("--debug", help="if you use debug mode", action="store_true")
    parser.add_argument("--load_model", help="Load model from path", default="")
    parser.add_argument("--seed", help="Seed for random number generator", default=42, type=int)
    parser.add_argument("--testmode", help="Test mode", action="store_true")
    args = parser.parse_args()

    if args.testmode:
        data_path = "../../../../../../mnt/Restricted/Corpora/CommonVoiceVTL/mini_corpus_" 
        args.data_path = data_path
        logging.info(f"Test mode: {args.data_path}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.optimizer == "adam":
        optimizer_module = torch.optim.Adam
    else:
        raise ValueError("Optimizer not supported")
    if args.criterion == "rmse":
        criterion = RMSELoss()
    else:
        raise ValueError("Criterion not supported")
    train_whole_dataset(
        data_path=args.data_path,
        skip_index=args.skip_index,
        start_epoch=args.start_epoch,
        optimizer_module=optimizer_module,
        criterion=criterion,
        language=args.language,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        validate_every=args.validate_every,
        save_every=args.save_every,
        load_from=args.load_model,
    )
    
