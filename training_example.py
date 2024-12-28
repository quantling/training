#!/usr/bin/env python
# coding: utf-8

#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/kaggle/input/vtl-data')
#cd /kaggle/input/vtl-data/

import pickle
import random
import time

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
tqdm.pandas()


from paule import models
import training
#import .testing


# cd /home/tino/ml_cloud_nextcloud/Common_Voice_data/

verbose = True

train = pd.read_pickle("common_voice_geco_words_train_slim_prot4.pkl")
valid = pd.read_pickle("common_voice_geco_words_valid_slim_prot4.pkl")
#train = pd.read_pickle("common_voice_oral_cavity_tube_norm_train.pkl")
#valid = pd.read_pickle("common_voice_oral_cavity_tube_norm_valid.pkl")
label_vectors = pd.read_pickle("lexical_embedding_vectors.pkl")
vectors = np.asarray(list(label_vectors.vector))

if verbose:
    print(train.columns)


column_input = "cp_norm"
column_output = "melspec_norm_synthesized"
#column_input = "tube_norm"
#column_output = "vector"

inps = train[column_input] 
tgts = train[column_output]
inps_valid = valid[column_input] 
tgts_valid = valid[column_output]


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed("20230925")


model = models.ForwardModel(num_lstm_layers=1,
                            hidden_size=720,
                            input_size=30,
                            output_size=60,
                            apply_half_sequence=True).double().to(DEVICE)

##  somatosensory pathway
#model = models.ForwardModel(num_lstm_layers=1,
#                            hidden_size=360,
#                            input_size=10,
#                            output_size=60,
#                            apply_half_sequence=True).double().to(DEVICE)
#
#
#model = models.EmbeddingModel(input_size=10,
#                              num_lstm_layers=2,
#                              hidden_size=360,
#                              dropout=0.7,
#                              post_upsampling_size=0).double()


model.train()
model.to(DEVICE)
if verbose:
    print(model)


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if verbose:
    print("Trainable Parameters in Model:", pytorch_total_params)


##  load pretrained weights
#with open('model_tube_to_vector_model_2_720_0_dropout_07_noise_6e05_rmse_lr_00001_250.pkl', 'rb') as pfile:
#    model, optimizer = pickle.load(pfile)
#model.to(DEVICE)

##  save model
#torch.save(model.state_dict(),"/kaggle/working/" + "model_tube_to_mel_model_1_360_lr_0001_50_00001_100.pt")



## reset learning rate in optimizer
#learning_rate = 0.0001
#for param_group in optimizer.param_groups:
#    print(param_group['lr'])
#    param_group['lr'] = learning_rate
#    print(param_group['lr'])



learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



criterion = training.RMSELoss(eps=0)
#criterion = training.cp_trjacetory_rmse_pos_vel_acc_jerk_loss


# In[11]:


num_epochs = 100
#continue_training_from = 250
continue_training_from = 2
batch_size = 8
seed = 24052022 #training.INV_FORW_SEED
save_model_after_i_iterations= 10

file_to_store = "cp_to_mel_model_1_720_0_dropout_07_noise_6e05_rmse_lr_00001" #"cp_to_tube_model_1_720_lr_0001_50_00001"#"cp_to_vector_model_2_720_0_dropout_07_noise_6e05_rmse_lr_00001
dict_file = "/home/tino/paule_training/results/"


# In[12]:


res_train = pd.DataFrame(columns=['epoch', 'loss', 'learning_rate'])
res_valid = pd.DataFrame(columns=['epoch', 'loss', 'learning_rate'])
#res_train = pd.DataFrame(columns=['epoch', 'loss', 'position_loss', 'velocity_loss', 'acceleration_loss', 'jerk_loss','learning_rate'])
#res_valid = pd.DataFrame(columns=['epoch', 'loss', 'position_loss', 'velocity_loss', 'acceleration_loss', 'jerk_loss','learning_rate'])
#res_train = pd.read_pickle("res_train_tube_to_vector_model_2_720_0_dropout_07_noise_6e05_rmse_lr_00001_250.pkl")
#res_valid = pd.read_pickle("res_valid_tube_to_vector_model_2_720_0_dropout_07_noise_6e05_rmse_lr_00001_250.pkl")



forward_training = training.Training(model, seed , inps, tgts, inps_valid, tgts_valid,
                 batch_size, res_train, res_valid, optimizer, criterion, use_same_size_batching=True)

embedder_training = training.Training(model, seed , 
                                     inps, tgts, inps_valid, tgts_valid,
                                     batch_size, res_train, res_valid, 
                                     optimizer, criterion, use_same_size_batching=True,
                                     use_cross_corr=False)



start_time = time.time()
forward_training.train(num_epochs=num_epochs, 
                       continue_training_from=continue_training_from,
                       file_to_store=file_to_store, 
                       dict_file=dict_file, 
                       save_model_after_i_iterations=save_model_after_i_iterations)
print(time.time() - start_time)



means = np.zeros(300)
stds = np.eye(300) * 6e-05


start_time = time.time()
embedder_training.train(num_epochs = num_epochs, 
                       continue_training_from=continue_training_from,
                       add_noise_to = "output",
                       mean_noise = means,
                       std_noise = stds,
                       fix_noise_per_epoch = False,
                       file_to_store = file_to_store, 
                       dict_file = dict_file, 
                       save_model_after_i_iterations = save_model_after_i_iterations)
print(time.time() - start_time)


# In[ ]:


# reset learning rate in optimizer
learning_rate = 0.000001
for param_group in optimizer.param_groups:
    print(param_group['lr'])
    param_group['lr'] = learning_rate
    print(param_group['lr'])



fig, ax = plt.subplots(figsize=(15, 10), facecolor="white")
tmp = res_train.groupby('epoch')['loss'].mean()
plt.semilogy(np.array(tmp.index), np.array(tmp), c='C0', lw=5, label='mean train loss')
del tmp
tmp = res_valid.iloc[1:].groupby('epoch')['loss'].mean()
plt.semilogy(np.array(tmp.index), np.array(tmp), c='C1', lw=5, label='mean validation loss')
ax.set_xlabel('Epoch', fontsize=20, labelpad=20)
#ax.set_ylabel('Loss (%s)' % str(loss_type), fontsize=20, labelpad=20)
ax.tick_params(axis='both', labelsize=15)
ax.legend(fontsize=15)
plt.savefig("loss.pdf")


torch.save(model.state_dict(), f"{file_to_store}_{num_epochs}.pt")
forward_training.res_train.to_pickle(f"{dict_file}res_train_{file_to_store}_{num_epochs}.pkl", protocol=4)
forward_training.res_valid.to_pickle(f"{dict_file}res_valid_{file_to_store}_{num_epochs}.pkl", protocol=4)

