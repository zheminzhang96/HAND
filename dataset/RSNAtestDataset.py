import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import v2

from torchvision import transforms
from torch.utils.data.dataset import Dataset


imgSize = 256

class RSNAtestDataset (Dataset):
    def __init__(self, df_test):
        self.df_test = df_test
        #self.traindir = traindir
        #self.imagenames = imagenames
        #self.labels = labels
        self.transformations = transforms.Compose([
                                     transforms.Resize((imgSize,imgSize)),
                                     transforms.ToTensor()
                                    ])
    def __getitem__(self, idx):
        img_in = Image.open(self.df_test['noise_path'].iloc[idx])
        img_in = img_in.convert('L')
        img_in = np.array(img_in)
        img_norm = (((img_in-np.min(img_in))/(np.max(img_in)-np.min(img_in)))*255).astype(dtype='uint8')

        # img_norm = np.expand_dims(img_norm, axis=-1)
        transformations = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Resize((imgSize,imgSize), antialias=True),
                                          transforms.Normalize(mean=[0.5], std=[0.5])])
        resize_img = transformations(img_norm)
        resize_img = resize_img.to(torch.float32)
        resize_img = (resize_img+1)/2

        return resize_img, self.df_test['label'].iloc[idx], self.df_test['noise'].iloc[idx], self.df_test['noise_path'].iloc[idx]
    
    def __len__(self): 
        return len(self.df_test)
