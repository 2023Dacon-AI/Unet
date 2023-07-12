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
import segmentation_models_pytorch as smp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
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

transform = A.Compose(
    [  
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2(),
    ]
)

train_transform = A.Compose(
    [      
        A.Resize(1024, 1024),    
        A.Normalize(),
        ToTensorV2(),
    ]
)

vaildation_transform = A.Compose(
    [        
        A.RandomCrop(224,224),  
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


class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False, is_validation=False, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer
        self.is_validation = is_validation
        if(is_test):
          pass
        elif(is_validation):
          self.data = self.data[6500:]
        else:
          self.data = self.data[:6500]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))
       
        if self.transform:
            if (self.is_validation):
                augmented = vaildation_transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                augmented = train_transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

        return image, mask


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
    
class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output
      
def train(num_epoch):
    # training loop
    for epoch in range(num_epoch):  # 10 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):   

            height, width = 1024, 1024
            crop_size = 224
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

        print(f'Epoch {epoch+1}, train_Loss: {epoch_loss/(len(dataloader)*25)} vali_loss: {vali_epoch_loss/len(vali_dataloader)} dice_score: {dice_score/len(vali_dataset)}')


def vali_inference():
  with torch.no_grad():
     model.eval()    
     Thresholds = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]
     for threshold in Thresholds:
        dice_score=0
        for images, masks in tqdm(vali_dataloader):
               images = images.float().to(device)
               masks = masks.float().to(device)

               outputs = model(images)

               preds = torch.sigmoid(outputs).detach().cpu().numpy()
               preds = np.squeeze(preds, axis=1)
               preds = (preds > threshold).astype(np.uint8) # Threshold = 0.35

               masks = masks.detach().cpu().numpy()

               dice_score+=calculate_dice_scores(masks,preds) * len(images)
        print(f"dice score for threshold {threshold}: {dice_score/len(vali_dataset)}")




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


# model 초기화

Encoder = 'timm-resnest101e' ## 46M 짜리 모델
Weights = 'imagenet'

prep_fun = smp.encoders.get_preprocessing_fn(
    Encoder,
    Weights
)

# model 초기화
model = smp.Unet(
    encoder_name = Encoder,
    encoder_weights = Weights,
    in_channels = 3,     
    classes = 1,
)

model.to(device)
# 저장한 모델 불러오기
model.load_state_dict(torch.load('unet_model_v8_25epoch.pt'))

# loss function과 optimizer 정의
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


if __name__ == '__main__':
    batch_size =64
    batch_sizes = 64 #46까지 하면 24GB 전부 사용
    epoch = 30

    dataset = SatelliteDataset(csv_file='./train.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    vali_dataset = SatelliteDataset(csv_file='./train.csv', transform=transform,is_validation=True)
    vali_dataloader = DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    #vali_inference()

    test_dataset = SatelliteDataset(csv_file='./test.csv', transform=transform, infer=True,is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    
    #train(epoch)  # 모델 훈련    
    test_inference()   
    # 모델 저장
    #torch.save(model.state_dict(),'unet_model_v5.pt')