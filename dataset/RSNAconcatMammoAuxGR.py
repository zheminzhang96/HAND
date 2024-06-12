import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from skimage.util import random_noise
import torchvision.transforms.functional as tf


imgSize = 256

class RSNAconcatMammoAuxDatasetGR (Dataset):
    def __init__(self, df_data):
        self.df_concat = df_data
        # self.label = flag
        #self.traindir = traindir
        #self.imagenames = imagenames
        #self.labels = labels
        self.transformations = transforms.Compose([
                                     transforms.Resize((imgSize,imgSize)),
                                     transforms.ToTensor()
                                    ])
    def __getitem__(self, i):
        img = Image.open(self.df_concat.iloc[i]['img_path'])
        img = img.convert('L')
        label = self.df_concat['label'].iloc[i]

        img = np.array(img)

        # normalize images
        img = (((img-np.min(img))/(np.max(img)-np.min(img)))*255).astype(dtype='uint8')


        transformations = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Resize((imgSize,imgSize), antialias=True),
                                            #transforms.RandomRotation(degrees=(-15,15)),
                                            transforms.Normalize(mean=[0.5], std=[0.5])])

        resize_img = transformations(img)

        resize_img = resize_img.to(torch.float32)
        resize_img = (resize_img+1)/2

        return resize_img, label
    
    def __len__(self): 
        #print("LEN", len(self.df_view['StudyInstanceUID'].unique()))
        return len(self.df_concat)