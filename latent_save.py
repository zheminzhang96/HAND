import os
import time
import random
import numpy as np
import pandas as pd
import torch
#from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from networks.TransBTS.TransBTS_aux import TransBTS
from dataset.build_dataset import *
import pickle

device = torch.device(f"cuda:{2}" if torch.cuda.is_available() else "cpu")
print("DEVICE INFO:", device)
_, model_bts = TransBTS(dataset='breast1', _conv_repr=True, _pe_type="learned")
checkpoint = torch.load('./checkpoint_aux/TransBTS2024-04-22/model_epoch_last.pth')
model_bts.load_state_dict(checkpoint['state_dict'])
# model_bts = load_ckpt('./checkpoint/TransBTS2024-04-10/model_epoch_last.pth', model_bts)
# model_bts.to(device)
model_bts.eval()


data_loader, data_size = build_breast_dataset(dataset_name='breast1', batch_size=4)
train_loader = data_loader['train']

train_latent = []
with torch.no_grad():
    for i, data in enumerate(train_loader):
        #images, _ = data
        images, labels, r_labels, n_labels, i_labels = data
        #print("images shape:", images.shape)
        for j in range(0, images.shape[0]):
            #print("images[j] shape:", images[j].shape)
            if labels[j] == 0 and r_labels[j] == 0 and n_labels[j] == 0 and i_labels[j] == 0:
                #print(labels[j], r_labels[j], n_labels[j], i_labels[j])
                x1_1, x2_1, x3_1, x, intmd_x = model_bts.encode(images[j].unsqueeze(0))
                train_latent.append(x)
                #print(x.shape)
    print("Appending done!")

with open('latent_train_aux.pkl', 'wb') as f:
    pickle.dump(train_latent, f)

