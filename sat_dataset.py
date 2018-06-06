##### ################################# #####
##### This code is written by Doi Kento #####
##### ################################# #####

import torch
import torch.utils.data as data_utils
import scipy.io as io
import numpy as np

class SAT_Dataset(data_utils.Dataset):
    """
    This dataset is for SAT-4 and SAT-6 dataset.
    Args:
        data_path (str): path of the dataset (.mat).
        phase     (str): phase (train, val or test).
    """
    def __init__(self, data_path, phase='train', transform=None):
        self.data = io.loadmat(data_path)
        if phase == 'train':
            self.image_arrays = self.data["train_x"][:,:,:,:300000].transpose(3,2,0,1)
            self.targets      = self.data["train_y"][:,:300000]
        elif phase == 'val':
            self.image_arrays = self.data["train_x"][:,:,:,300000:].transpose(3,2,0,1)
            self.targets      = self.data["train_y"][:,300000:]
            
        else:
            self.image_arrays = self.data["test_x"].transpose(3,2,0,1)
            self.targets      = self.data["test_y"]

        self.transform = transform

    def __getitem__(self, idx):
        img = self.image_arrays[idx,:,:,:]

        if self.transform:
            img = self.transform(img)

        return [img, self.targets[:, idx] == 1)[0][0]]

    def __len__(self):
        return self.targets.shape[1]
