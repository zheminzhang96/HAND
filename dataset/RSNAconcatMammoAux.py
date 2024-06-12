import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from skimage.util import random_noise
import torchvision.transforms.functional as tf


imgSize = 256

class RSNAconcatMammoAuxDataset (Dataset):
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
        img = Image.open(self.df_concat.iloc[i]['img_path'])
        img = img.convert('L')
        label = self.df_concat['label'].iloc[i]
        rotate_label = 0
        invert_label = 0
        colorJ_label = 0
        mask_label = 0
        natural_label = 0

        noise_gen = random.randint(0, 1)
        noise_int = 0
        #print("generate noise or not:", noise_gen)

        if noise_gen == 1:
            noise_int = random.randint(1, 5)
            #print("noise int random:", noise_int)

        if noise_int == 1:
            randome_angle = random.uniform(-50, 50)
            img = tf.rotate(img=img, angle=randome_angle)
            rotate_label = 1
        if noise_int == 2:
            img = tf.invert(img=img)
            invert_label = 1
        if noise_int == 3 :
            colorJ = transforms.ColorJitter(brightness=(2.0, 2.5), contrast=(0.7, 1.0), saturation=(0.5, 0.7), hue=0.5)
            img = colorJ(img)
            colorJ_label = 1
        if noise_int == 5:
            natural_p = select_random_image('./natural_tr/')
            #print(natural_p)
            img = resize_and_paste(original_image=img, random_image_path=natural_p)
            natural_label = 1

        img = np.array(img)

        # normalize images
        img = (((img-np.min(img))/(np.max(img)-np.min(img)))*255).astype(dtype='uint8')


        transformations = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Resize((imgSize,imgSize), antialias=True),
                                            #transforms.RandomRotation(degrees=(-15,15)),
                                            transforms.Normalize(mean=[0.5], std=[0.5])])
        # print("RSNA img shape before transform:", img.shape)
        resize_img = transformations(img)

        resize_img = resize_img.to(torch.float32)
        resize_img = (resize_img+1)/2
        if noise_int == 4:
            resize_img = apply_round_mask(resize_img, mask_size=0.3)
            mask_label = 1
            # print("max pixel:", torch.max(resize_img))
            # print("min pixel:", torch.min(resize_img))
        return resize_img, label, rotate_label, invert_label, colorJ_label, mask_label, natural_label
    
    def __len__(self): 
        #print("LEN", len(self.df_view['StudyInstanceUID'].unique()))
        return len(self.df_concat)

def apply_round_mask(image, mask_size):
    _, h, w = image.size()

    # Calculate the radius of the circular mask (20% of the smaller dimension)
    radius = min(h, w) * mask_size / 2

    # Create a grid of coordinates
    x = torch.arange(w).float().view(1, -1)
    y = torch.arange(h).float().view(-1, 1)
    
    mask = torch.zeros(1, h, w)
    center = (h//2, w//2)
    #y, x= torch.ogrid[:h, :w]
    mask[0, (x-center[1])**2+(y-center[0])**2 <= radius**2] = 1
    #print("mask:", mask)
    mask = 1-mask
    print("mask shape:", mask.shape)

    masked_image = image.clone()
    masked_image[ mask==0] = 0.7

    return masked_image

def select_random_image(folder_path):
    # List all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('png'))]
    # Choose a random image from the list
    random_image = random.choice(image_files)
    return os.path.join(folder_path, random_image)

def resize_and_paste(original_image, random_image_path):
    x_pos = random.randint(0, 120)
    y_pos = random.randint(0, 120)
    print((x_pos, y_pos))
    random_image = Image.open(random_image_path)
    random_image = random_image.convert('L')
    random_image_resized = random_image.resize((120, 120))
    
    # Paste the resized random image onto the original image
    original_image.paste(random_image_resized, (x_pos, y_pos))
    
    return original_image