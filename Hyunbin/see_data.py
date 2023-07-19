import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Dataset import SatelliteDataset
import matplotlib.pyplot as plt

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


def check_image(dataset, start, num):
    for i in range(start,start+num):
      img, mask = dataset[i]
      imshow(img, mask,i)

transform = A.Compose(
    [
        A.Normalize(),
        ToTensorV2()
    ]
)

checking_dataset = SatelliteDataset(csv_file='./train.csv', transform=transform)
num=128
check_image(checking_dataset, 1321, num)