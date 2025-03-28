import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


class AccedingSequenceLengthBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size, drop_last=False):
        # Get the lengths of sequences
        self.sizes = data_source.sizes # Assuming cp_norm has the sequence lengths
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        indices = torch.argsort(torch.tensor(self.sizes)).tolist()
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches.pop()
        # Shuffle the batches
        np.random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sizes) // self.batch_size
        else:
            return (len(self.sizes) + self.batch_size - 1) // self.batch_size
    
def pad_tensor(tensor, target_length, allow_longer = False):
    """Pads the tensor to target_length by repeating the last element.
    Returns a mask """
    if not isinstance(tensor, torch.Tensor):	
        logging.error(f"tensor: {tensor}")
        raise ValueError("Input tensor must be a torch.Tensor")
    current_length = tensor.shape[0]
    if current_length > target_length and not allow_longer:
        raise ValueError(f" {target_length}, {current_length}") # if we don't have max size as target sths wrong
    if current_length == target_length:
        return tensor, torch.ones(target_length, dtype=torch.bool)

    logging.debug(f"tensor shape: {tensor.shape}")
    #logging.debug(f"tensor: {tensor}")
    last_element = tensor[-1].unsqueeze(0)  # Get the last element
    padding = last_element.repeat(target_length - current_length, *[1] * (tensor.dim() - 1))
    mask = torch.cat([
    torch.ones(current_length, dtype=torch.bool),
    torch.zeros(target_length - current_length, dtype=torch.bool)
])
    return torch.cat([tensor, padding], dim=0), mask



def plot_validation_losses(validation_losses, language, save_path=".", model_name="forward_model"):
    """
    Plots the mean validation losses over epochs and saves the plot.
    
    :param validation_losses: List of mean validation losses.
    :param language: String representing the language.
    :param save_path: Directory where the plot should be saved.
    """
    epochs = range(1, len(validation_losses) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, validation_losses, marker='o', linestyle='-', label='Mean Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Validation Loss over Epochs ({language})')
    plt.legend()
    plt.grid(True)
    
    plot_name = f"validation_losses_{model_name}_{language}_{len(validation_losses)}.png"
    plt.savefig(f"{save_path}/{plot_name}")
    plt.show()
    
    print(f"Plot saved as {save_path}/{plot_name}")



def validate_whole_dataset(files, data_path, batch_size = 8, device = DEVICE, criterion = None,  model = None, validate_on_one_df = None):
    
    if model is None:
        raise ValueError("Model is not defined")
    if validate_on_one_df is None:
        raise ValueError("validate_on_one_df is not defined")
    logging.debug(files)
    total_losses = []
    for file in files:
        logging.info(f" Model type {model}")
        mean_loss, std_loss, epoch_losses =validate_on_one_df(
            batch_size=batch_size,
            device=device,
            file_path=os.path.join(data_path, file),
            criterion=criterion,
            model=model,
        )
        total_losses.extend(epoch_losses)
        logging.info(f"Mean loss: {mean_loss}, Std loss: {std_loss}")

        gc.collect()
    
    with open("validation_losses.txt", "w") as f:
        for loss in total_losses:
            f.write(f"{loss}\n")
    
    return np.mean(total_losses), np.std(total_losses)