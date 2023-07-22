import os

import cv2
import pandas as pd
from torch.utils.data import Dataset

from common.utils import rle_decode


class SatelliteDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, val=False, test=False):
        self.data_dir = data_dir
        self.data = pd.read_csv(os.path.join(data_dir, csv_file))
        self.transform = transform
        self.val = val
        self.test = test

        val_index = [11, 37, 107, 194, 276, 311, 387, 417, 543, 614, 682, 802, 813, 870, 892, 896, 1112, 1113, 1127, 1275, 1308, 1414, 1427, 1449, 1450, 1549, 1655, 1787, 2033, 2093, 2126, 2248, 2271, 2370, 2470, 2554,2600, 2652, 2680, 2708, 2844, 2917, 2919, 2953, 2955, 3227, 3288, 3312, 3460, 3489, 3559, 3608, 3735, 3780, 3874, 3884, 3910, 3981, 4056, 4060, 4132, 4174, 4360, 4448, 4632, 4682, 4777, 4801, 4883, 4887, 4938, 4956, 4999, 5074, 5097, 5101, 5121, 5152, 5173, 5327, 5445, 5534, 5568, 5630, 5634, 5722, 5897, 5957, 6195, 6197, 6265, 6384, 6457, 6470, 6484, 6521, 6536, 6590, 6720, 6742]
        train_index = [id for id in range(len(self.data)) if id not in val_index]

        if(self.val): 
          self.data = self.data.iloc[val_index]
        else:
          self.data = self.data.iloc[train_index]


    def __len__(self):
      if self.test:
        return len(self.data)
      else:
        return len(self.data) * 25
      

    def processing(self,idx, height=1024, width=1024, crop_size=200): #crop_size=200 -> 개수 맞추기 위함

      xy = idx//len(self.data) # 0~24의 값
      randx = xy//5 # 0~4의 값
      randy = xy%5 # 0~4의 값

      idx = idx%len(self.data) #25배로 불린 이유는 crop을 위함, 다시 원래대로

      img_path = os.path.join(self.data_dir, self.data.iloc[idx, 1])
      image = cv2.imread(img_path)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      if(self.test):
        return image

      mask_rle = self.data.iloc[idx, 2]
      mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

      left = crop_size * (randx)
      top = crop_size * (randy)

      right = left + 224
      bottom = top + 224

      cropped_image = image[top:bottom, left:right, :]
      cropped_mask = mask[top:bottom, left:right]

      return cropped_image, cropped_mask


    def __getitem__(self, idx):
        if (self.test):
            image = self.processing(idx)
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        image, mask = self.processing(idx)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        batch = dict(
            image=image,
            mask=mask,
        )

        return batch
