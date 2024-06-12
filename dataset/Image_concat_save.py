import pandas as pd
import numpy as np
import os
import pickle as pkl
import re
import matplotlib.pyplot as plt
from skimage import io
import skimage.io
import skimage.measure as skmeas
import sys
import numpy as np
import logging
import numpy as np
import csv
from RSNAbreastCancer import *
#from .RSNAtestDataset import *

from torchvision import transforms
from torch.utils.data import DataLoader
import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import image_preprocessing as image_preprocessing

# Run the code: python3 ./dataset/Image_concat_save.py

train_df_mlo, test_df_mlo = get_breast_data('MLO')
train_df_cc, test_df_cc = get_breast_data('CC')
train_df_mlo.to_csv("./dataset/train_df_mlo.csv")
#val_df_mlo.to_csv("./dataset/val_df_mlo.csv")
test_df_mlo.to_csv("./dataset/test_df_mlo.csv")
train_df_cc.to_csv("./dataset/train_df_cc.csv")
#val_df_cc.to_csv("./dataset/val_df_cc.csv")
test_df_cc.to_csv("./dataset/test_df_cc.csv")

def image_process(df_view, i, view):
    #print("i in __getitem__",i)
    df_cc_groups = df_view.groupby(['StudyInstanceUID'])
    df_cc_group_i = df_cc_groups.get_group(df_view['StudyInstanceUID'].unique()[i])
    # 
    if len(df_cc_group_i[df_cc_group_i['ImageLaterality']=='R']) == 1:
        img_r = Image.open(df_cc_group_i[df_cc_group_i['ImageLaterality']=='R']['png_path'].iloc[0]) #get and open the png path with laterality R 
        #print(df_cc_group_i[df_cc_group_i['ImageLaterality']=='R']['png_path'].iloc[0])
    else:
        return
    if len(df_cc_group_i[df_cc_group_i['ImageLaterality']=='L']) == 1:
        img_l = Image.open(df_cc_group_i[df_cc_group_i['ImageLaterality']=='L']['png_path'].iloc[0]) #get and open the png path with laterality L
        #print(df_cc_group_i[df_cc_group_i['ImageLaterality']=='L']['png_path'].iloc[0])
    else:
        return
    img_r = np.array(img_r)
    img_l = np.array(img_l)
    # print("min img_r:", np.min(img_r))
    # print("max img_r", np.max(img_r))
    # print("min img_l:", np.min(img_l))
    # print("max img_l", np.max(img_l))

    # normalize images
    img_r = (((img_r-np.min(img_r))/(np.max(img_r)-np.min(img_r)))*255).astype(dtype='uint8')
    img_l = (((img_l-np.min(img_l))/(np.max(img_l)-np.min(img_l)))*255).astype(dtype='uint8')

    if df_cc_group_i[df_cc_group_i['ImageLaterality']=='R']['PhotometricInterpretation'].iloc[0]=='MONOCHROME1': #invert white background to black background
        #print("INVERT")
        img_r = np.invert(img_r)
    if df_cc_group_i[df_cc_group_i['ImageLaterality']=='L']['PhotometricInterpretation'].iloc[0]=='MONOCHROME1': #invert white background to black background
        #print("INVERT")
        img_l = np.invert(img_l)

    #print(img.shape)
    if check_laterality(img_r, 'R') == True:
        img_crop_r = image_preprocessing.segment_breast(img_r)     
    else:
        img_crop_r = image_preprocessing.segment_breast(img_l)
        
    if check_laterality(img_l, 'L') == True:
        img_crop_l = image_preprocessing.segment_breast(img_l)
    else:
        img_crop_l = image_preprocessing.segment_breast(img_r)
        
    img_conct = image_preprocessing.stitch_images(img_crop_r, img_crop_l)

    #print("noise type:", df_cc_group_i['noise'].iloc[0])
    # if df_cc_group_i['noise'].iloc[0] == 'gaussian':
    #     img_conct = random_noise(np.array(img_conct), mode='gaussian', var=0.1)
    #     img_conct = np.array(255*img_conct, dtype='uint8')
    transforms_img = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((256, 256), antialias= True)])
    img_conct = transforms_img(img_conct)
    img_conct = np.array(img_conct)
    img_conct = img_conct.reshape([256, 256])
    img_conct = (((img_conct-np.min(img_conct))/(np.max(img_conct)-np.min(img_conct)))*255).astype(dtype='uint8')
    
    pixel_mean = int(np.average(img_conct))
    pixel_std = int(np.std(img_conct))
    # min_p_count = sum(1 for row in img_conct for p in row if p < 2)
    # max_p_count = sum(1 for row in img_conct for p in row if p > 253)
    min_p_count = np.sum(img_conct < 2)
    max_p_count = np.sum(img_conct > 253)
    #min_p_count 
    img_path = '/home/zhemin/Datacenter_storage/zhemin/mammo_concat_png/'+str(df_cc_group_i['PatientID'].iloc[0])+view+'.png'
    plt.imsave(img_path, img_conct, cmap='gray')
    
    return df_cc_group_i['PatientID'].iloc[0], df_cc_group_i['StudyInstanceUID'].iloc[0], df_cc_group_i['label'].iloc[0], df_cc_group_i['noise'].iloc[0], pixel_mean, pixel_std, min_p_count, max_p_count, img_path

def concat_csv(df_in, view):
    column_names = ['patient_id', 'study_id', 'label', 'noise', 'pixel_mean', 'pixel_std', 'min_p_count', 'max_p_count', 'img_path', 'view']
    df_concat = pd.DataFrame(columns=column_names)
    for i in range(len(df_in['StudyInstanceUID'].unique())):
        p_id, s_id, label, noise, pixel_mean, pixel_std, min_count, max_count, path = image_process(df_in, i, view)
        df_concat.loc[i] = [p_id, s_id, label, noise, pixel_mean, pixel_std, min_count, max_count, path, view]
    return df_concat

if __name__ == '__main__':
    # print("Training dataset size:", len(train_df_cc)/2+len(train_df_mlo)/2)
    # print("Start generating training CC concatenated png and csv...")
    # train_df_cc_concat = concat_csv(train_df_cc, 'CC')
    # train_df_cc_concat.to_csv('/home/zhemin/Datacenter_storage/zhemin/mammo_concat_csv/train_cc_concat.csv') 
    # print("Start generating training MLO concatenated png and csv...")
    # train_df_mlo_concat = concat_csv(train_df_mlo, 'MLO')
    # train_df_mlo_concat.to_csv('/home/zhemin/Datacenter_storage/zhemin/mammo_concat_csv/train_mlo_concat.csv')

    print("Testing dataset size:", len(test_df_cc)/2+len(test_df_mlo)/2)
    print("Start generating testing CC concatenated png and csv...")
    test_df_cc_concat = concat_csv(test_df_cc, 'CC')
    test_df_cc_concat.to_csv('/home/zhemin/Datacenter_storage/zhemin/mammo_concat_csv/test_cc_concat.csv')
    print("Start generating testing MLO concatenated png and csv...")
    test_df_mlo_concat = concat_csv(test_df_mlo, 'MLO')
    test_df_mlo_concat.to_csv('/home/zhemin/Datacenter_storage/zhemin/mammo_concat_csv/test_mlo_concat.csv')