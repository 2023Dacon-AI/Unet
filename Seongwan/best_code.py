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
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# RLE 디코딩 함수
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


# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()
    
transform = A.Compose(
    [  
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2(),
    ]
)

vaildation_transform = A.Compose(
    [   
        A.HorizontalFlip(p=0.5),  # 좌우 대칭
        A.VerticalFlip(p=0.5),    # 상하 대칭
        A.RandomCrop(224,224),  
        A.Normalize(),
        ToTensorV2(),
    ]
)

train_transform = A.Compose(
    [    
        A.HorizontalFlip(p=0.5),  # 좌우 대칭
        A.VerticalFlip(p=0.5),    # 상하 대칭  
        A.Resize(1024, 1024),    
        A.Normalize(),
        ToTensorV2(),
    ]
)


def dice_score(prediction: np.array, ground_truth: np.array, smooth=1e-7) -> float:
    '''
    Calculate Dice Score between two binary masks.
    '''
    intersection = np.sum(prediction * ground_truth)
    return (2.0 * intersection + smooth) / (np.sum(prediction) + np.sum(ground_truth) + smooth)


def calculate_dice_scores(ground_truth_df, prediction_df, img_shape=(224, 224)) -> List[float]:
    '''
    Calculate Dice scores for a dataset.
    '''


    # Keep only the rows in the prediction dataframe that have matching img_ids in the ground truth dataframe
    #prediction_df = prediction_df[prediction_df.iloc[:, 0].isin(ground_truth_df.iloc[:, 0])]
    #prediction_df.index = range(prediction_df.shape[0])


    # Extract the mask_rle columns
    #pred_mask_rle = prediction_df.iloc[:, 1]
    #gt_mask_rle = ground_truth_df.iloc[:, 1]
    pred_mask_rle = prediction_df
    gt_mask_rle = ground_truth_df


    def calculate_dice(pred_rle, gt_rle):
        #pred_mask = rle_decode(pred_rle, img_shape)
        #gt_mask = rle_decode(gt_rle, img_shape)
        pred_mask = pred_rle
        gt_mask = gt_rle

        if np.sum(gt_mask) > 0 or np.sum(pred_mask) > 0:
            return dice_score(pred_mask, gt_mask)
        else:
            return None  # No valid masks found, return None


    dice_scores = Parallel(n_jobs=-1)(
        delayed(calculate_dice)(pred_rle, gt_rle) for pred_rle, gt_rle in zip(pred_mask_rle, gt_mask_rle)
    )


    dice_scores = [score for score in dice_scores if score is not None]  # Exclude None values
    return np.mean(dice_scores)

def cutmix(image1, image2, mask1,mask2):
    # 이미지 크기 및 채널 수 확인
    _, height, width = image1.shape

    # 랜덤하게 가로, 세로 위치 선정
    cut_height = int(height * random.uniform(0.2, 0.8))
    cut_width = int(width * random.uniform(0.2, 0.8))

    # 가운데 위치 선정
    start_height = int((height - cut_height) / 2) + int(height * random.uniform(-0.2,0.2))
    start_width = int((width - cut_width) / 2) + int(width * random.uniform(-0.2,0.2))

    # 이미지 자르기
    image1_cut = image1[:,start_height:start_height+cut_height, start_width:start_width+cut_width]
    image2_cut = image2[:,start_height:start_height+cut_height, start_width:start_width+cut_width]

    # CutMix 적용
    mixed_image = np.copy(image1)
    mixed_image[:,start_height:start_height+cut_height, start_width:start_width+cut_width] = image2_cut


    mask1_cut = mask1[start_height:start_height+cut_height, start_width:start_width+cut_width]
    mask2_cut = mask2[start_height:start_height+cut_height, start_width:start_width+cut_width]

    # CutMix 적용
    mixed_mask = np.copy(mask1)
    mixed_mask[ start_height:start_height+cut_height, start_width:start_width+cut_width] = mask2_cut


    return mixed_image, mixed_mask


class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False, is_validation=False, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer
        self.is_validation = is_validation
        self.validation_index = [11, 37, 107, 194, 276, 311, 387, 417, 543, 614, 682, 802, 813, 870, 892, 896, 1112, 1113, 1127, 1275, 1308, 1414, 1427, 1449, 1450, 1549, 1655, 1787, 2033, 2093, 2126, 2248, 2271, 2370, 2470, 2554,2600, 2652, 2680, 2708, 2844, 2917, 2919, 2953, 2955, 3227, 3288, 3312, 3460, 3489, 3559, 3608, 3735, 3780, 3874, 3884, 3910, 3981, 4056, 4060, 4132, 4174, 4360, 4448, 4632, 4682, 4777, 4801, 4883, 4887, 4938, 4956, 4999, 5074, 5097, 5101, 5121, 5152, 5173, 5327, 5445, 5534, 5568, 5630, 5634, 5722, 5897, 5957, 6195, 6197, 6265, 6384, 6457, 6470, 6484, 6521, 6536, 6590, 6720, 6742]
        self.train_index = [id for id in range(len(self.data)) if id not in self.validation_index]

        if(is_test): # test일 때
          pass
        elif(self.is_validation): # validation일 때
          self.data = self.data.iloc[self.validation_index]
        else: #train일 때
          self.data = self.data.iloc[self.train_index]

    def __len__(self):
        if(self.is_validation):
          return 5*len(self.data)
        else:
          return len(self.data)

    def __getitem__(self, idx):

        if(self.is_validation):
          idx = idx % len(self.data)

        img_path = self.data.iloc[idx, 2]
        image = cv2.imread(img_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)       
        if self.infer:
            if self.transform: #infer는 test_transform으로
                image = transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 3]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
          if self.is_validation: #validation
            augmented = vaildation_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

          else: #train
                augmented = train_transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

        return image, mask
    
def train(num_epoch):
    # training loop
    for epoch in range(num_epoch):  # 10 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):   
            height, width = 1024, 1024
            crop_size = 224
            # 이미지 224*224로 자르기
            for top in range(0, height, crop_size): 
                for left in range(0, width, crop_size):

                    right = left + crop_size
                    bottom = top + crop_size
                    cropped_image = images[:,:,top:bottom, left:right]
                    cropped_mask = masks[:, top:bottom, left:right]

                    cropped_image = cropped_image.float().to(device)
                    cropped_mask = cropped_mask.float().to(device)

                    optimizer.zero_grad()
                    outputs = model(cropped_image)
                    loss = criterion(outputs, cropped_mask.unsqueeze(1))
                    loss.backward()
                    optimizer.step()
        
                    epoch_loss += loss.item()
        # lr 조정           
        scheduler.step()
        print("lr: ", optimizer.param_groups[0]['lr'])
        model.eval()
        vali_epoch_loss=0
        dice_score = 0

        for images, masks in tqdm(vali_dataloader):
            images = images.float().to(device)
            masks = masks.float().to(device)

            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = np.squeeze(preds, axis=1)
            preds = (preds > 0.5).astype(np.uint8) # Threshold = 0.35

            vali_epoch_loss += loss.item()

            masks = masks.detach().cpu().numpy()

            dice_score+=calculate_dice_scores(masks,preds) * len(images)

            torch.save(model.state_dict(),f'unet_model_v10_{epoch+1}epoch.pt')

        # train 이미지가 25배가 되었음으로 나눠줘야 함.
        print(f'Epoch {epoch+1}, train_Loss: {epoch_loss/(len(dataloader)*25)} vali_loss: {vali_epoch_loss/len(vali_dataloader)} dice_score: {dice_score/len(vali_dataset)}')


def vali_inference():
  with torch.no_grad():
     model.eval()
     dice_score=0
     for images, masks in tqdm(vali_dataloader):
            images = images.float().to(device)
            masks = masks.float().to(device)

            outputs = model(images)

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = np.squeeze(preds, axis=1)
            preds = (preds > 0.5).astype(np.uint8) # Threshold = 0.35

            masks = masks.detach().cpu().numpy()

            dice_score+=calculate_dice_scores(masks,preds) * len(images)
     print(f"dice score for validation: {dice_score/len(vali_dataset)}")


def test_inference():
    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)

            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.5).astype(np.uint8) # Threshold = 0.35

            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if len(mask_rle.split()) < 10: # 예측된 건물 픽셀이 아예 없는 경우 -1
                    result.append(-1)
                else:
                    result.append(mask_rle)
        submit = pd.read_csv('./sample_submission.csv')
        submit['mask_rle'] = result
        submit.to_csv('./submit.csv', index=False)


Encoder = 'timm-resnest101e' ## 46M 짜리 모델
Weights = 'imagenet'

prep_fun = smp.encoders.get_preprocessing_fn(
    Encoder,
    Weights
)

# model 초기화
model = smp.UnetPlusPlus(
    encoder_name = Encoder,
    encoder_weights = Weights,
    in_channels = 3,     
    classes = 1,
)

model.to(device)
# 저장한 모델 불러오기
#model.load_state_dict(torch.load('unet_model_v9_17epoch.pt'))

# loss function과 optimizer 정의
criterion = MixedLoss(alpha = 10.0,gamma = 2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.9 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)

if __name__ == '__main__':
    batch_size = 14 # 현재 모델은 36이 VRAM 24GB로 최대임
    epoch = 30
    dataset = SatelliteDataset(csv_file='./train_remove_little_damaged.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    vali_dataset = SatelliteDataset(csv_file='./train_remove_little_damaged.csv', transform=transform, is_validation=True)
    vali_dataloader = DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    test_dataset = SatelliteDataset(csv_file='./test.csv', transform=transform, infer=True, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    
    train(epoch)  # 모델 훈련
    vali_inference()
    #test_inference()
    # 모델 저장
    torch.save(model.state_dict(),'unet_model_v10_fianl.pt')