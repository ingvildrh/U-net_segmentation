import os 
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

'''
Class to create an object of the dataset
'''
class DriveDataset(Dataset):
    '''
    Init function to initialize the dataset
    INPUT: 
        images_path : paths to the images
        masks_path : paths to the corresponding masks
    '''
    def __init__(self, images_path, masks_path):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    '''
    Function to get the mask and the image corresponding to an index
    INPUT:
        index : index of the image
    OUTPUT:
        image : image corresponding to the index in the dataset
        mask : mask corresponding to the index in the dataset
    '''
    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask
    '''
    Function to get the length of the dataset
    OUTPUT:
        length : length of the dataset
    '''
    def __len__(self):
        return self.n_samples