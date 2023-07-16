import os
import cv2
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import List, Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import RandomCrop, Compose
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import random


def denoramlize(img):
    img = img.permute(1,2,0)            # change shape ---> (width, height, channel)
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])
    img = img*std + mean
    img = np.clip(img,0,1)              # convert the pixel values range(min=0, max=1)
    return img

def imshow(img, mask,i):
    fig = plt.figure(figsize=(20, 15))
    a = fig.add_subplot(1, 3, 1)
    plt.imshow(denoramlize(img), cmap='bone')   
    a.set_title(f"Original image {i}")
    plt.grid(False)
    plt.axis("off")
  
    if(mask!=None):     
      a = fig.add_subplot(1, 3, 2)
      plt.imshow(mask, cmap='binary')      
      a.set_title("The mask")
      plt.grid(False)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)      
      plt.axis("off")      
        
      a = fig.add_subplot(1, 3, 3)
      img[:, mask == 1] = 255
      plt.imshow(denoramlize(img), cmap='bone')      
      a.set_title("The mask + img")
      plt.grid(False)
      plt.xticks([])
      plt.yticks([])
      plt.axis("off")
      plt.grid(False)
      plt.show()


def check_image(dataset, start):
    for i in range(start,start+1398):
      img, mask = dataset[i]
      imshow(img, mask,i)
      



transform = A.Compose(
    [
        A.Normalize(),
        ToTensorV2()
    ]
)

def rle_decode(mask_rle: Union[str, int], shape=(224, 224)) -> np.array:
    '''
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    if mask_rle == -1:
        return np.zeros(shape)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
          return len(self.data)

    def __getitem__(self, idx):

        cutmix_idx = random.randint(0,len(self.data)-1)

        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

checking_dataset = SatelliteDataset(csv_file='./train.csv', transform=transform)

check_image(checking_dataset,5643)