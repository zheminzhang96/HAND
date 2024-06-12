import logging
import numpy as np
import csv

#from dataset.RSNAbreastCancer import *
from dataset.RSNAconcatMammo import *
from dataset.RSNAconcatMammoAux import *
from dataset.RSNAconcatMammoAux2 import *
from dataset.RSNAconcatMammoAuxGR import *
from dataset.RSNAconcatMammoAuxGR_noise import *
from sklearn.model_selection import train_test_split


from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

# cifar10_tsfms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

def build_breast_dataset(dataset_name, batch_size):
    #logger = logging.getLogger()
    print("Build RSNA breast dataset for {}".format(dataset_name))
    
    assert dataset_name in ['breast', 'breast1', 'breast2']
        

    if dataset_name == "breast": # this load concatenated breast png 
        train_df = pd.read_csv('/home/zhemin/Datacenter_storage/zhemin/mammo_concat_csv/train_s3.csv')
        # train_df_cc = pd.read_csv ('/home/zhemin/Datacenter_storage/zhemin/mammo_concat_csv/train_cc_s2.csv')
        
        train_set = RSNAconcatMammoDataset(train_df)
        
        print("Total training size: ", len(train_set))

    elif dataset_name == "breast1": # this load concatenated breast png with aux branches
        train_df = pd.read_csv('/home/zhemin/Datacenter_storage/zhemin/mammo_concat_csv/train_s3.csv')
        # train_df_cc = pd.read_csv ('/home/zhemin/Datacenter_storage/zhemin/mammo_concat_csv/train_cc_s2.csv')

        #train_set = RSNAconcatMammoAuxDataset(train_df) # output 6 labels for each augmentation
        train_set = RSNAconcatMammoAuxDataset2(train_df) # output 1 label for augmented or not
        
        print("**********************************************************")
        print("Total training size: ", len(train_set))

    elif dataset_name == 'breast2': # this load concatenated breast png with aux branches with GR
        train_df = pd.read_csv('/home/zhemin/Datacenter_storage/zhemin/mammo_concat_csv/train_s3.csv')
        #train_id, train_ood = train_test_split(train_df, test_size=0.5, random_state=1996)
        
        #train_id_set = RSNAconcatMammoAuxDatasetGR(train_id) # output 1 label for augmented or not
        train_set = RSNAconcatMammoAuxDatasetGR_noise(train_df)

        print("**********************************************************")
        print("Total training size: ", len(train_set))


    ## add test set later: 'test': DataLoader(test_set, batch_size = cvae_batch_size, num_workers = 1)
    ## , 'test':len(test_set)
    # breast_dataloaders = {'train': DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)}
    #                       #'val': DataLoader(validate_set, batch_size = cvae_batch_size, num_workers = 1)}
    # breast_dataset_sizes = {'train': len(train_set)}

    breast_dataloaders = {'train': DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)}
                          #'trainOOD': DataLoader(train_ood_set, batch_size = batch_size, num_workers = 8)}
    breast_dataset_sizes = {'train': len(train_set)}
        
    return breast_dataloaders, breast_dataset_sizes
