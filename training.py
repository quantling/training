"""
This module defines a class to do the training of the component models of
PAULE.

"""


import os
import pickle
import random
import time

import pandas as pd
import numpy as np
import torch
from torch.nn import L1Loss
from torch.nn import MSELoss

#from utils import *

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        if shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter

if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
tqdm.pandas()

INV_FORW_SEED = "11012021"
INV_FORW_NOISE_SEED = 20203003
EMBEDDER_SEED = "02012021"

########################################################################################################################
###################################### Helper Functions ################################################################
########################################################################################################################

def corrcoef(x):
    """
    Mimics `np.corrcoef`

    :param x: 2D torch.Tensor

    :return c: 2D torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)
    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref:
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013
    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c

def get_vel_acc_jerk(trajectory, *, lag=1):
    """
    Approximate the velocity, acceleration and jerk of an input trajectory batch per sample and feature

    :param trajectory: 3D torch.Tensor (batch, seq_length, features)
        input trajectory
    :param lag: int
        lag offset
    :return velocity, acc, jerk: 3D torch.Tensors
        approximated velocities, acc, jerk per sample in batch and channel
    """
    #velocity = trajectory[:, lag:, :] - trajectory[:, :-lag, :]
    #acc = velocity[:, 1:, :] - velocity[:, :-1, :]
    #jerk = acc[:, 1:, :] - acc[:, :-1, :]

    velocity = (trajectory[:, lag:, :] - trajectory[:, :-lag, :]) / lag
    acc = (velocity[:, 1:, :] - velocity[:, :-1, :]) / 1
    jerk = (acc[:, 1:, :] - acc[:, :-1, :]) / 1
    return velocity, acc, jerk


def pad_to_max_length(xx, max_len,with_onset_dim = False):
    """
    Pad array of trajectories to a maximal length
    - pad at the end with last value

    :param xx: np.array
        trajectories (seq_length, features)
    :param max_len: int
        length to be padded
    :param with_onset_dim: bool
        add a new feature with 1 for the first time step rest 0 as an indicator for the onset
    :return: 2D torch.Tensor
        padded trajectory (max_len, features) or if with_onset_dim = True (max_len, features + 1)
    """
    seq_length = xx.shape[0]
    if with_onset_dim:
        onset = np.zeros((seq_length, 1))
        onset[0, 0] = 1
        xx = np.concatenate((xx, onset), axis=1) # shape len X (features +1)
    padding_size = max_len - seq_length
    xx = np.concatenate((xx, np.tile(xx[-1:, :],(padding_size,1))), axis=0)
    return torch.from_numpy(xx)


def add_and_pad(xx, max_len, with_onset_dim=False, with_time_dim = False):
    """
    Padd a sequence with last value to maximal length

    :param xx: 2D np.array
        seuence to be padded (seq_length, feeatures)
    :param max_len: int
        maximal length to be padded to
    :param with_onset_dim: bool
        add one features with 1 for the first time step and rest 0 to indicate sound onset
    :return: 2D torch.Tensor
        padded sequence
    """
    if len(xx.shape)>1:
        seq_length = xx.shape[0]
    else:
        seq_length = 1
    if with_onset_dim:
        onset = np.zeros((seq_length, 1))
        onset[0, 0] = 1
        xx = np.concatenate((xx, onset), axis=1)  # shape len X (features +1)
    if with_time_dim:
        time_dim = np.linspace(0, 1, num=seq_length).reshape(-1,1)[::-1]
        xx = np.concatenate((xx, time_dim), axis=1)
    padding_size = max_len - seq_length
    padding_size = tuple([padding_size] + [1 for i in range(len(xx.shape) - 1)])
    xx = np.concatenate((xx, np.tile(xx[-1:], padding_size)), axis=0)
    return torch.from_numpy(xx)


def pad_batch_online(lens, data_to_pad, device = "cpu", with_onset_dim=False, with_time_dim=False):
    """
    :param lens: 1D torch.Tensor
        Tensor containing the length of each sample in data_to_pad of one batch
    :param data_to_pad: series
        series containing the data to pad
    :return padded_data: torch.Tensors
        Tensors containing the padded and stacked to one batch
    """
    max_len = int(max(lens))
    padded_data = torch.stack(list(data_to_pad.apply(lambda x: add_and_pad(x, max_len,with_onset_dim=with_onset_dim, with_time_dim=with_time_dim)))).to(device)

    return padded_data



########################################################################################################################
###################################### Loss Functions ##################################################################
########################################################################################################################

l1 = L1Loss()
l2 = MSELoss()

class RMSELoss(torch.nn.Module):
    """
    Root-Mean-Squared-Error-Loss (RMSE-Loss)
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

rmse = RMSELoss(eps=0)

def cp_trjacetory_rmse_pos_vel_acc_jerk_loss(Y_hat, tgts, apply_l1_regularization_derivatives = False):
    """
    Calculate additive loss using the RMSE of position velocity , acc and jerk

    :param Y_hat: 3D torch.Tensor
        model prediction
    :param tgts: 3D torch.Tensor
        target tensor
    :param apply_l1_regularization_derivatives: bool
        apply l1 regularization on the derivatives in order to enforce model predictions resulting in sparse derivatives
    :return loss, pos_loss, vel_loss, acc_loss, jerk_loss: torch.Tensors
            loss, pos_loss, vel_loss, acc_loss, jerk_loss, vel_loss_stationary, acc_loss_stationary, jerk_loss_stationary:  torch.Tensors
        summed total loss with all individual losses
    """

    velocity, acc, jerk = get_vel_acc_jerk(tgts)
    velocity2, acc2, jerk2 = get_vel_acc_jerk(tgts, lag=2)
    velocity4, acc4, jerk4 = get_vel_acc_jerk(tgts, lag=4)

    Y_hat_velocity, Y_hat_acceleration, Y_hat_jerk = get_vel_acc_jerk(Y_hat)
    Y_hat_velocity2, Y_hat_acceleration2, Y_hat_jerk2 = get_vel_acc_jerk(Y_hat, lag=2)
    Y_hat_velocity4, Y_hat_acceleration4, Y_hat_jerk4 = get_vel_acc_jerk(Y_hat, lag=4)

    pos_loss = rmse(Y_hat, tgts)
    vel_loss = rmse(Y_hat_velocity, velocity) + rmse(Y_hat_velocity2, velocity2) + rmse(Y_hat_velocity4, velocity4)
    jerk_loss = rmse(Y_hat_jerk, jerk) + rmse(Y_hat_jerk2, jerk2) + rmse(Y_hat_jerk4, jerk4)
    acc_loss = rmse(Y_hat_acceleration, acc) + rmse(Y_hat_acceleration2, acc2) + rmse(Y_hat_acceleration4, acc4)

    if apply_l1_regularization_derivatives:
        vel_loss_stationary = l1(Y_hat_velocity, torch.zeros_like(Y_hat_velocity)) + \
                              l1(Y_hat_velocity2,torch.zeros_like(Y_hat_velocity2)) + \
                              l1(Y_hat_velocity4, torch.zeros_like(Y_hat_velocity4))
        acc_loss_stationary = l1(Y_hat_acceleration,torch.zeros_like(Y_hat_acceleration)) + \
                              l1(Y_hat_acceleration2,torch.zeros_like(Y_hat_acceleration2)) + \
                              l1(Y_hat_acceleration4,torch.zeros_like(Y_hat_acceleration4))
        jerk_loss_stationary = l1(Y_hat_jerk,torch.zeros_like(Y_hat_jerk)) + \
                               l1(Y_hat_jerk2,torch.zeros_like(Y_hat_jerk2)) + \
                               l1(Y_hat_jerk4,torch.zeros_like(Y_hat_jerk4))

        loss = pos_loss + vel_loss + acc_loss + jerk_loss + vel_loss_stationary + acc_loss_stationary + jerk_loss_stationary
        return loss, pos_loss, vel_loss, acc_loss, jerk_loss, vel_loss_stationary, acc_loss_stationary, jerk_loss_stationary

    loss = pos_loss + vel_loss + acc_loss + jerk_loss
    return loss, pos_loss, vel_loss, acc_loss, jerk_loss


def cross_corr_loss(Y_hat_cross_corr, tgts_cross_corr, Y_hat, tgts):
    """
    Calculate the cross correlation loss by using the RMSE of the correlations between the predicted and all other true semantic vectors

    :param Y_hat_cross_corr: 2D torch.Tensor
        correlation between predicted vectors in batch and all other semantic vectors
    :param tgts_cross_corr: 2D torch.Tensor
        true correlation between vectors in batch and all other semantic vectors
    :param Y_hat: 2D torch.Tensor
        predicted semantic vectors
    :param tgts: 2D torch.Tensor
        true semantic vectors
    :param reg_lambda: regularization weighting
    :return loss, cross_corr_loss, rmse_loss: torch.Tensor
        summed total loss with all individual losses
    """

    cross_corr_loss = rmse(Y_hat_cross_corr, tgts_cross_corr)
    rmse_loss = rmse(Y_hat, tgts)
    loss = cross_corr_loss
    return loss, cross_corr_loss, rmse_loss


def regularized_cross_corr_loss(Y_hat_cross_corr, tgts_cross_corr, Y_hat, tgts, reg_lambda = 1):
    """
    Calculate additive loss using the RMSE of the correlations between the predicted and all other true semantic vectors and the RMSE of the
    predicted and true semantic vectors

    :param Y_hat_cross_corr: 2D torch.Tensor
        correlation between predicted vectors in batch and all other semantic vectors
    :param tgts_cross_corr: 2D torch.Tensor
        true correlation between vectors in batch and all other semantic vectors
    :param Y_hat: 2D torch.Tensor
        predicted semantic vectors
    :param tgts: 2D torch.Tensor
        true semantic vectors
    :param reg_lambda: regularization weighting
    :return loss, cross_corr_loss, rmse_loss: torch.Tensor
        summed total loss with all individual losses
    """

    cross_corr_loss = rmse(Y_hat_cross_corr, tgts_cross_corr)
    rmse_loss = rmse(Y_hat, tgts)
    loss = cross_corr_loss + reg_lambda * rmse_loss
    return loss, cross_corr_loss, rmse_loss

def nll_loss(pred_means_stds, tgts):
    """
    Calculate Negative Log Likelihood Loss

    :param pred_means_stds: 3D torch.Tensor (batch, seq_length, means of features + std of features )
        predicted means and stds per timestep and feature
    :param tgts: 3D torch.Tensor (batch, seq_length, featurees)
        target outputs
    :return:
        negative log likelihood of targets given the predicted means and stds
    """
    # we must return a scalar as that what pytorch requires for backpropagation
    means = pred_means_stds[:,:,:60]
    stds = torch.clamp(pred_means_stds[:,:,60:], min=1e-6)
    dist = torch.distributions.Normal(means, stds)
    return -dist.log_prob(tgts).sum()


def get_decomposition_matrix(cov):
    try:
        return np.linalg.cholesky(cov), "cholesky"
    except np.linalg.LinAlgError as e:
        return np.linalg.svd(cov), "SVD"


def sample_multivariate_normal(mean, decomposition_matrix, decomposition):
    if decomposition == "cholesky":
        standard_normal_vector = np.random.standard_normal(len(decomposition_matrix))
        return decomposition_matrix @ standard_normal_vector + mean
    if decomposition == "SVD":
        u, s, vh = decomposition_matrix
        standard_normal_vector = np.random.standard_normal(len(u))
        return u @ np.diag(np.sqrt(s)) @ vh @ standard_normal_vector + mean
    raise ValueError("use either ' chelesky' or 'SVD' as decomposition")





########################################################################################################################
################################################ Training ##############################################################
########################################################################################################################

class Training:
    """
    Create Training Instance
    :param model: torch model
        model to train
    :param seed: int
        int to set seed in order to be reproducible
    :param inps_train: pd.Series
        series containing the inputs of the training set
    :param tgts_train: pd.Series
        series containing the corresponding targets
    :param inps_valid: pd.Series
        series containing the inputs of the validation set
    :param tgts_valid: pd.Series
        series containing the corresponding targets

    :param batch_size: int
        batch size
    :param res_train: pd.DataFrame
        pd.DataFrame for logging epoch results on training set
    :param res_valid: pd.DataFrame
        pd.DataFrame for logging epoch results on validation set

    :param optimizer: torch.optim.Optimizer
        torch optimizer for updating weights of the model
    :param criterion: torch.Loss
        criterion to calculate the loss between targets and predictions
        - if using cross correlation the function must accept predicted correlations vs. true correlation and predicted vectors vs. target vectors

    :param use_same_size_batching: bool
        specify whether to batch inps with similar length during epoch creating in order to avoid long padding

    :param use_cross_corr: bool
        specify whether to calculate the cross correlation

    (necessary for calculating cross correlation)
    :param labels_train: pd.Series
        series containing the label for each input sample in the training set
    :param labels_valid: pd.Series
        series containing the label for each input sample in the validation set
    :param cross_corr_matrix: pd.DataFrame
        pd.DataFrame containing the pairwise correlation between all unique labels
        - labels are used and row and column index
    :param  label_vectors: pd.DataFrame
        pd.DataFrame containing the semantic vectors for each label
        - labels are used as row index

    """
    def __init__(self, model, seed, inps_train, tgts_train,
                 inps_valid, tgts_valid, batch_size,
                 res_train, res_valid,
                 optimizer, criterion,
                 use_same_size_batching=True,
                 with_onset_dim=False,
                 with_time_dim=False,
                 use_cross_corr=False,
                 labels_train=(), labels_valid=(),
                 cross_corr_matrix = None, label_vectors = None):

        self.seed = seed
        self.model = model
        self.inps_train = inps_train
        self.tgts_train = tgts_train
        self.inps_valid = inps_valid
        self.tgts_valid = tgts_valid

        self.input_dimension = inps_train.iloc[0].shape[-1]
        self.output_dimension = tgts_train.iloc[0].shape[-1]

        self.res_train = res_train
        self.res_train_ix = len(res_train)
        self.res_valid = res_valid
        self.res_valid_ix = len(res_valid)

        self.device = next(model.parameters()).device # get device model is located at

        # get lengths of inputs and outputs
        if len(inps_train.iloc[0].shape) > 1:
            self.lens_input = torch.tensor(np.array(inps_train.apply(len), dtype=int)).to(self.device)
            self.lens_input_valid = torch.tensor(np.array(inps_valid.apply(len), dtype=int)).to(self.device)
        else:
            self.lens_input = torch.ones(len(inps_train),dtype=int).to(self.device)
            self.lens_input_valid = torch.ones(len(inps_valid),dtype=int).to(self.device)
        if len(tgts_train.iloc[0].shape) > 1:
            self.lens_output = torch.tensor(np.array(tgts_train.apply(len), dtype=int)).to(self.device)
            self.lens_output_valid = torch.tensor(np.array(tgts_valid.apply(len), dtype=int)).to(self.device)
        else:
            self.lens_output = torch.ones(len(tgts_train),dtype=int).to(self.device)
            self.lens_output_valid = torch.ones(len(tgts_valid),dtype=int).to(self.device)




        self.batch_size = batch_size
        self.optimizer = optimizer
        self.criterion = criterion

        self.use_cross_corr = use_cross_corr

        if use_cross_corr:
            assert len(labels_train)>0 and len(labels_valid)>0, "In order to use cross correlation please provide valid training and validation labels!"
            assert not cross_corr_matrix is None, "Please provide a precomputed Cross Correlation Matrix DataFrame!"
            assert not label_vectors is None, "Please provide a lookup DataFrame with labels and corresponding embedding vectors!"
            self.labels_train = labels_train
            self.labels_valid = labels_valid
            self.cross_corr_matrix = cross_corr_matrix
            self.label_vectors = label_vectors
            self.label_vectors_torch = torch.from_numpy(np.asarray(list(label_vectors.vector))).to(self.device)

        self.use_same_size_batching = use_same_size_batching
        self.with_onset_dim = with_onset_dim
        self.with_time_dim = with_time_dim

        # is using same size batching we create a dictionary containing all unique lengths and the indices of each sample with a this length
        if self.use_same_size_batching:
            self.train_length_dict = {}
            lengths, counts = np.unique(self.lens_input.cpu(), return_counts=True)
            self.sorted_length_keys = np.sort(lengths)

            for length in self.sorted_length_keys:
                self.train_length_dict[length] = np.where(self.lens_input.cpu() == length)[0]

    def create_epoch_batches(self, df_length, batch_size, shuffle=True, same_size_batching=False):
        """
        Create Epoch by batching indices

        :param df_length: int
            total number of samples in training set
        :param batch_size: int
            number of samples in one batch
        :param shuffle: bool
            keep order of training set or random shuffle
        :param same_size_batching: bool
            create epoch of batches with similar long samples to avoid long padding
        :return epoch: list of list
            list of lists containing indices for each batch in one epoch
        """
        if same_size_batching:
            epoch = [] # list of batches
            foundlings = []  # rest samples for each length which do not fit into one batch
            for length in self.sorted_length_keys: # iterate over each unique length in training data
                length_idxs = self.train_length_dict[length] # dictionary containing indices of samples with length
                rest = len(length_idxs) % batch_size
                random.shuffle(length_idxs) # shuffle indices
                epoch += [length_idxs[i * batch_size:(i * batch_size) + batch_size] for i in
                          range(int(len(length_idxs) / batch_size))] # cut into batches and append to epoch
                if rest > 0:
                    foundlings += list(length_idxs[-rest:]) # remaining indices which do not fit into one batch are stored in foundling
            foundlings = np.asarray(foundlings)
            rest = len(foundlings) % batch_size
            epoch += [foundlings[i * batch_size:(i * batch_size) + batch_size] for i in
                      range(int(len(foundlings) / batch_size))] # cut foudnlings into batches (because inserted sorted this ensures minimal padding)
            if rest > 0:
                epoch += [foundlings[-rest:]] # put rest into one batch (allow smaller batch)
            random.shuffle(epoch)

        else:
            rest = df_length % batch_size
            idxs = list(range(df_length))
            if shuffle:
                random.shuffle(idxs) # shuffle indicees
            if rest > 0:
                idxs += idxs[:(batch_size - rest)] # rolling batching (if number samples not divisible by batch_size append first again)
            epoch = [idxs[i * batch_size:(i * batch_size) + batch_size] for i in range(int(len(idxs) / batch_size))] # cut into batches

        return epoch


    def evaluate(self):
        """
        Function for evaluating model on validation set

        :return valid_predictions, valid_losses, (valid_subplosses):
            - predictions made by the model for each sample in validation set
            - the loss for each sample with respect to the provided criterion
            - (if loss contains sublosses each individual loss)
        """

        with torch.no_grad():
            valid_predictions = []
            valid_losses = []
            valid_sublosses = []

            with torch.no_grad():  # no gradient calculation
                epoch_test = self.create_epoch_batches(len(self.inps_valid), 1, shuffle=False)
                for jj, idxs in enumerate(tqdm(epoch_test, desc='Evaluating on Validation Set')):
                    lens_input_jj = self.lens_input_valid[idxs]
                    batch_input = self.inps_valid.iloc[idxs]
                    batch_input = pad_batch_online(lens_input_jj, batch_input,self.device,self.with_onset_dim, self.with_time_dim)

                    lens_output_jj = self.lens_output_valid[idxs]

                    Y_hat = self.model(batch_input,lens_input_jj)

                    batch_output = self.tgts_valid.iloc[idxs]
                    batch_output = pad_batch_online(lens_output_jj, batch_output, self.device)
                    batch_output = torch.squeeze(batch_output,1)

                    if self.use_cross_corr:
                        batch_output_cross_corr = self.labels_valid.iloc[idxs]
                        batch_output_cross_corr = torch.from_numpy(np.asarray(self.cross_corr_matrix.loc[batch_output_cross_corr])).to(self.device)
                        Y_hat_cross_corr = corrcoef(torch.cat((Y_hat, self.label_vectors_torch)))[:1,1:]
                        loss = self.criterion(Y_hat_cross_corr, batch_output_cross_corr, Y_hat, batch_output)
                    else:
                        loss = self.criterion(Y_hat, batch_output)

                    if isinstance(loss, tuple): # sublosses
                        sub_losses = loss[1:]  # rest sublosses
                        loss = loss[0] # first total loss

                        valid_losses += [loss.item()]
                        valid_sublosses += [[sub_loss.item() for sub_loss in sub_losses]] # for each samples [subloss1_i,subloss2_i,subloss3_i]

                    else:
                        valid_losses += [loss.item()]

                    prediction = Y_hat.cpu().detach().numpy()[0]
                    valid_predictions +=[prediction]

                if len(valid_sublosses) > 0:
                    test_sublosses = np.asarray(valid_sublosses)
                    return valid_predictions, valid_losses, \
                           [test_sublosses[:, i] for i in range(test_sublosses.shape[1])] # for each subloss [subloss1_i, subloss1_j,subloss1_k]

                return valid_predictions, valid_losses, []



    def train(self, num_epochs, continue_training_from, shuffle=True,
              add_noise_to = None,
              mean_noise = None,
              std_noise = None,
              fix_noise_per_epoch = False,
              validate_after_epoch=1,
              verbose=True,
              save_model_after_i_iterations=1,
              dict_file = "",
              file_to_store= time.strftime("%Y%m%d-%H%M%S")):
        """
        Train the model

        :param num_epochs: int
            number of epochs to train
        :param continue_training_from: int
            epoch to resume training from
        :param shuffle: bool
            whether to use shuffle in creating epochs
        :param validate_after_epoch: int
            number of epochs to train before evaluate model
        :param verbose: bool
            print results after each epoch
        :param save_model_after_i_iterations: int
            number of epochs to train model before saving it
        :param dict_file: str
            dictionary to store the model in
        :param file_to_store: str
            name of files to store model, training and validation results
        """

        if add_noise_to:
            assert add_noise_to in ["input", "output"], "Please specify whether to add noise to 'input' or 'output'"
            if add_noise_to == "input":
                assert len(mean_noise) == self.input_dimension, "Please provide a valid Mean vector of the N-dimensional distribution"
                assert len(std_noise) == self.input_dimension, "Please provide a valid Covariance matrix of the N-dimensional distribution"
                if fix_noise_per_epoch:
                    assert len(torch.unique(self.lens_input)) == 1, "Fixed noise only possible for constant input length"
                    if len(self.inps_train.iloc[0].shape) >1:
                        constant_length = int(np.unique(self.lens_input)[0])
                    else:
                        constant_length = 1

            else:
                assert len(mean_noise) == self.output_dimension, "Please provide a valid Mean vector of the N-dimensional distribution"
                assert len(std_noise) == self.output_dimension, "Please provide a valid Covariance matrix of the N-dimensional distribution"
                if fix_noise_per_epoch:
                    assert len(torch.unique(self.lens_output)) == 1, "Fixed noise only possible for constant output length"
                    if len(self.tgts_train.iloc[0].shape) >1:
                        constant_length = int(np.unique(self.lens_output)[0])
                    else:
                        constant_length = 1

            decomposition_matrix, decomposition = get_decomposition_matrix(std_noise)

        random.seed(self.seed)
        np.random.seed(int(self.seed))

        if not os.path.isdir(dict_file):
            os.mkdir(dict_file)


        if continue_training_from > 0: # ensure continue with same batch
            epoch_valid = self.create_epoch_batches(len(self.inps_valid), 1, shuffle=False) #
            for i in range(continue_training_from):
                epoch = self.create_epoch_batches(len(self.inps_train), self.batch_size, shuffle=shuffle,same_size_batching=self.use_same_size_batching)
                epoch_valid = self.create_epoch_batches(len(self.inps_valid), self.batch_size, shuffle=False)
                if add_noise_to:
                    if fix_noise_per_epoch:
                        noise = np.asarray([sample_multivariate_normal(mean_noise, decomposition_matrix, decomposition) for x in range(constant_length)])
                        #noise = np.random.multivariate_normal(mean_noise,std_noise,constant_length)
                    else:
                        if add_noise_to == "input":
                            for idxs in epoch:
                                lens_input_jj = self.lens_input[idxs]
                                #noise = [np.random.multivariate_normal(mean_noise,std_noise,int(l)) for l in lens_input_jj]
                                noise = [np.asarray([sample_multivariate_normal(mean_noise, decomposition_matrix, decomposition) for x in range(l)]) for l in lens_input_jj]
                        else:
                            for idxs in epoch:
                                lens_output_jj = self.lens_output[idxs]
                                #noise = [np.random.multivariate_normal(mean_noise,std_noise,int(l)) for l in lens_output_jj]
                                noise = [np.asarray([sample_multivariate_normal(mean_noise, decomposition_matrix, decomposition) for x in range(l)]) for l in lens_output_jj]


        else:
            _, validation_losses, validation_sublosses = self.evaluate() # inital validation loss
            if len(validation_sublosses) > 0:
                average_epoch_valid_sublosses = [np.mean(sub_loss) for sub_loss in validation_sublosses] # calculate mean sublosses
                self.res_valid.loc[self.res_valid_ix] = [-1, np.mean(validation_losses)] + average_epoch_valid_sublosses + [param_group["lr"] for param_group in self.optimizer.param_groups]
            else:
                average_epoch_valid_sublosses = []
                self.res_valid.loc[self.res_valid_ix] = [-1, np.mean(validation_losses)] + [param_group["lr"] for param_group in self.optimizer.param_groups]
            self.res_valid_ix += 1

            if verbose:
                print("\nInitial Validation Loss: ", np.mean(validation_losses))
                if len(average_epoch_valid_sublosses) > 0:
                    for i, subloss in enumerate(average_epoch_valid_sublosses):
                        print("Subloss %d: " % i, subloss)

        for ii in tqdm(range(num_epochs), desc ='Training...',position=0,leave=True):
            ii += continue_training_from
            average_epoch_loss = []
            average_epoch_sublosses = []

            running_loss = 0.0
            epoch = self.create_epoch_batches(len(self.inps_train), self.batch_size,same_size_batching=self.use_same_size_batching)

            if add_noise_to:
                if fix_noise_per_epoch:
                    #noise = np.random.multivariate_normal(mean_noise,std_noise,constant_length)
                    noise = np.asarray([sample_multivariate_normal(mean_noise, decomposition_matrix, decomposition) for x in range(constant_length)])


            for jj, idxs in enumerate(tqdm(epoch,desc = "Batch...",position=1, leave=False)): #enumerate(epoch):
                # index by indices in batch
                lens_input_jj = self.lens_input[idxs]
                batch_input = self.inps_train.iloc[idxs].copy()

                lens_output_jj = self.lens_output[idxs]
                batch_output = self.tgts_train.iloc[idxs].copy()

                if add_noise_to:
                    if add_noise_to == "input":
                        if fix_noise_per_epoch:
                            batch_noise = pd.Series(list(np.tile(noise,(len(batch_input),1))))
                            batch_noise.index = batch_input.index
                            batch_input += batch_noise
                        else:
                            batch_input_index = batch_input.index
                            #batch_input = pd.Series([batch_input.iloc[i] + np.random.multivariate_normal(mean_noise,std_noise,int(l)) for i,l in enumerate(lens_input_jj)])
                            batch_input = pd.Series([batch_input.iloc[i] + np.asarray([sample_multivariate_normal(mean_noise, decomposition_matrix, decomposition) for x in range(l)]) for i,l in enumerate(lens_input_jj)])
                            batch_input.index = batch_input_index
                    else:
                        if fix_noise_per_epoch:
                            batch_noise = pd.Series(list(np.tile(noise,(len(batch_output),1))))
                            batch_noise.index = batch_output.index
                            batch_output += batch_noise
                        else:
                            batch_output_index = batch_output.index
                            #batch_output = pd.Series([batch_output.iloc[i] + np.random.multivariate_normal(mean_noise,std_noise,int(l)) for i,l in enumerate(lens_output_jj)])
                            batch_output = pd.Series([batch_output.iloc[i] + np.asarray([sample_multivariate_normal(mean_noise, decomposition_matrix, decomposition) for x in range(l)]) for i,l in enumerate(lens_output_jj)])
                            batch_output.index = batch_output_index

                batch_input = pad_batch_online(lens_input_jj, batch_input,self.device,self.with_onset_dim, self.with_time_dim)
                batch_output = pad_batch_online(lens_output_jj, batch_output,self.device)

                Y_hat = self.model(batch_input,lens_input_jj)

                batch_output = torch.squeeze(batch_output,1)

                self.optimizer.zero_grad() # set gradients to zero before performing backprop
                if self.use_cross_corr:
                    batch_output_cross_corr = self.labels_train.iloc[idxs]
                    batch_output_cross_corr = torch.from_numpy(np.asarray(self.cross_corr_matrix.loc[batch_output_cross_corr])).to(self.device)
                    Y_hat_cross_corr = corrcoef(torch.cat((Y_hat, self.label_vectors_torch)))[:len(batch_output_cross_corr), len(batch_output_cross_corr):]
                    loss = self.criterion(Y_hat_cross_corr, batch_output_cross_corr, Y_hat, batch_output)
                else:
                    loss = self.criterion(Y_hat, batch_output)

                if isinstance(loss, tuple):
                    sub_losses = loss[1:]  # sublosses
                    loss = loss[0] # total loss

                    average_epoch_loss += [loss.item()]
                    average_epoch_sublosses += [[sub_loss.item() for sub_loss in sub_losses]] # for each batch [subloss1_i,subloss2_i,subloss3_i]

                else:
                    average_epoch_loss += [loss.item()]

                running_loss += loss.item()
                loss.backward()  # compute dloss/dx and accumulated into x.grad
                self.optimizer.step()  # compute x += -learning_rate * x.grad

            if len(average_epoch_sublosses) > 0:
                average_epoch_sublosses = np.asarray(average_epoch_sublosses)
                average_epoch_sublosses = [average_epoch_sublosses[:, i] for i in
                                           range(average_epoch_sublosses.shape[1])]  # for each subloss [subloss1_i, subloss1_j,subloss1_k]
                average_epoch_sublosses = [np.mean(sub_loss) for sub_loss in average_epoch_sublosses]
                self.res_train.loc[self.res_train_ix] = [ii, np.mean(average_epoch_loss)] + average_epoch_sublosses + [param_group["lr"] for param_group in self.optimizer.param_groups]
            else:
                average_epoch_sublosses = []
                self.res_train.loc[self.res_train_ix] = [ii, np.mean(average_epoch_loss)] + [param_group["lr"] for param_group in self.optimizer.param_groups]

            self.res_train_ix += 1

            if verbose:
                print("\nAvg Training Loss: ", np.mean(average_epoch_loss))
                print("Running Training Loss: ", float(running_loss))
                if len(average_epoch_sublosses) > 0:
                    for i, subloss in enumerate(average_epoch_sublosses):
                        print("Subloss %d: " % i, subloss)

            ########################################################
            ###################### Validation ######################
            ########################################################

            if (ii+1) % validate_after_epoch == 0:
                _, validation_losses, validation_sublosses = self.evaluate()
                if len(validation_sublosses) > 0:
                    average_epoch_valid_sublosses = [np.mean(sub_loss) for sub_loss in validation_sublosses]
                    self.res_valid.loc[self.res_valid_ix] = [ii, np.mean(
                        validation_losses)] + average_epoch_valid_sublosses + [param_group["lr"] for param_group in
                                                                               self.optimizer.param_groups]
                else:
                    average_epoch_valid_sublosses = []
                    self.res_valid.loc[self.res_valid_ix] = [ii, np.mean(validation_losses)] + [param_group["lr"] for param_group in self.optimizer.param_groups]
                self.res_valid_ix += 1

                if verbose:
                    print("\nAvg Validation Loss ", np.mean(validation_losses))
                    if len(average_epoch_valid_sublosses) > 0:
                        for i, subloss in enumerate(average_epoch_valid_sublosses):
                            print("Subloss %d: " % i, subloss)

                if not save_model_after_i_iterations is None:
                    if ii > 0 and (ii+1) % save_model_after_i_iterations == 0:
                        self.res_train.to_pickle(dict_file + "/res_train_" + file_to_store + "_%d" % (ii+1) + ".pkl")
                        self.res_valid.to_pickle(dict_file + "/res_valid_" + file_to_store + "_%d" % (ii+1) + ".pkl")
                        with open(dict_file + "/model_" + file_to_store + "_%d" % (ii+1) + ".pkl", "wb") as pfile:
                            pickle.dump((self.model, self.optimizer), pfile)
