import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from common.utils import rle_decode
from torch.utils.data import Dataset


class SatelliteDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, val=False, test=False):
        self.data_dir = data_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.val = val
        self.test = test

        val_index = [11, 37, 107, 194, 276, 311, 387, 417, 543, 614, 682, 802, 813, 870, 892, 896, 1112, 1113, 1127, 1275, 1308, 1414, 1427, 1449, 1450, 1549, 1655, 1787, 2033, 2093, 2126, 2248, 2271, 2370, 2470, 2554,2600, 2652, 2680, 2708, 2844, 2917, 2919, 2953, 2955, 3227, 3288, 3312, 3460, 3489, 3559, 3608, 3735, 3780, 3874, 3884, 3910, 3981, 4056, 4060, 4132, 4174, 4360, 4448, 4632, 4682, 4777, 4801, 4883, 4887, 4938, 4956, 4999, 5074, 5097, 5101, 5121, 5152, 5173, 5327, 5445, 5534, 5568, 5630, 5634, 5722, 5897, 5957, 6195, 6197, 6265, 6384, 6457, 6470, 6484, 6521, 6536, 6590, 6720, 6742]
        train_index = [id for id in range(len(self.data)) if id not in val_index]
        
        if test: pass
        elif val: self.data = self.data.iloc[val_index]
        else: self.data = self.data.iloc[train_index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data.iloc[idx, 1])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.test:
            if self.transform: 
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
          if self.val:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
          else:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        batch = dict(
            image=image,
            mask=mask,
        )

        return batch
    

transform = A.Compose(
    [
        A.RandomCrop(height=224, width=224, always_apply=True),
        A.Normalize(),
        ToTensorV2()
    ]
)


class ModifiedSatelliteDataset(Dataset):
    def __init__(self, data_dir, csv_file, augmentation=None, preprocessing=None, infer=False):
        self.data_dir = data_dir
        self.data = pd.read_csv(csv_file)
        self.augmentation = augmentation
        self.infer = infer
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data.iloc[idx, 1])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.augmentation:
                image = self.augmentation(image=image)['image']

            if self.preprocessing:
                image = self.preprocessing(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))
        new_mask = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=np.uint8)
        new_mask[:, :, 0] = 1 - mask
        new_mask[:, :, 1] = mask
        mask = new_mask


        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask



def get_training_augmentation():
    train_transform = [    
        A.RandomCrop(height=224, width=224, always_apply=True),
        A.OneOf(
            [
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():   
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        A.PadIfNeeded(min_height=224, min_width=224, always_apply=True, border_mode=0),
    ]
    return A.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(A.Lambda(image=preprocessing_fn))
    _transform.append(A.Lambda(image=to_tensor, mask=to_tensor))
        
    return A.Compose(_transform)