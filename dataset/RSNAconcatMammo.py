import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from skimage.util import random_noise


imgSize = 256

class RSNAconcatMammoDataset (Dataset):
    def __init__(self, df_data):
        self.df_concat = df_data
        #self.traindir = traindir
        #self.imagenames = imagenames
        #self.labels = labels
        self.transformations = transforms.Compose([
                                     transforms.Resize((imgSize,imgSize)),
                                     transforms.ToTensor()
                                    ])
    def __getitem__(self, i):
        # print("i in __getitem__", i)
        # print(i.dtype)
        img = Image.open(self.df_concat.iloc[i]['img_path'])
        img = img.convert('L')
        img = np.array(img)
  
        # normalize images
        img = (((img-np.min(img))/(np.max(img)-np.min(img)))*255).astype(dtype='uint8')

        # if self.df_concat['noise'].iloc[i] == 'gaussian':
        #     img_noise = random_noise(np.array(img), mode='gaussian', var=0.01)
        #     img = np.array(255*img_noise, dtype='uint8')
        # if self.df_concat['noise'].iloc[i] == 'salt_pepper':
        #     img = random_noise(np.array(img), mode='s&p', amount=0.6)
        #     img = np.array(255*img, dtype='uint8')
        # if self.df_concat['noise'].iloc[0] == 'distort':
        #     img = distort_img(img)
        
        transformations = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Resize((imgSize,imgSize), antialias=True),
                                          transforms.RandomRotation(degrees=(-15,15)),
                                          transforms.Normalize(mean=[0.5], std=[0.5])])
        # print("RSNA img shape before transform:", img.shape)
        resize_img = transformations(img)

        
        resize_img = resize_img.to(torch.float32)
        resize_img = (resize_img+1)/2
        # print("RSNA img shape:", resize_img.shape)
        # print('max pixel value:', torch.max(resize_img))
        # print('min pixel value:', torch.min(resize_img))
        return resize_img, self.df_concat['label'].iloc[i]
    
    def __len__(self): 
        #print("LEN", len(self.df_view['StudyInstanceUID'].unique()))
        return len(self.df_concat)

def distort_img(image):
    roll_img = np.array(image)
    A = roll_img.shape[0] / 3.0
    w = 2.0 / roll_img.shape[1]
    shift = lambda x: A * np.sin(2.0*np.pi*x * w)
    for i in range(roll_img.shape[0]):
        roll_img[:,i] = np.roll(roll_img[:,i], int(shift(i)))

    return roll_img