import torch
import numpy as np
import segmentation_models_pytorch as smp
from MixedLoss import MixedLoss
from Dataset import SatelliteDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from dicescore import calculate_dice_scores
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_dir = '/content/drive/MyDrive/'
model_name='unet_'

# Model
Encoder = 'timm-resnest269e'
Weights = 'imagenet'
prep_fun = smp.encoders.get_preprocessing_fn(
    Encoder,
    Weights
)

model = smp.Unet(
    encoder_name = Encoder,
    encoder_weights = Weights,
    in_channels = 3,
    classes=1,
    aux_params=dict(
        pooling='max',             # one of 'avg', 'max'
    )
)
model.to(device)

criterion = MixedLoss(alpha = 10.0,
                      gamma = 2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=3, verbose=True
)

#Transform
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),  # 좌우 대칭
        A.VerticalFlip(p=0.5),    # 상하 대칭
        #A.RandomCrop(224,224),
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)
transform_val = A.Compose(
    [
        A.HorizontalFlip(p=0.5),  # 좌우 대칭
        A.VerticalFlip(p=0.5),    # 상하 대칭
        #A.RandomCrop(224,224),
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)

batch_size=40
epochs=80

dataset = SatelliteDataset(csv_file='./train.csv', transform=transform, val=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

dataset_val = SatelliteDataset(csv_file='./train.csv', transform=transform_val, val=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

best_dice_score=0

for epoch in range(epochs): 
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader):
        images = images.float().to(device)
        masks = masks.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.unsqueeze(1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    model.eval()
    epoch_loss_val=0
    dice_score = 0
    for images, masks in tqdm(dataloader_val):
        images = images.float().to(device)
        masks = masks.float().to(device)

        outputs = model(images)
        loss = criterion(outputs, masks.unsqueeze(1))

        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = np.squeeze(preds, axis=1)
        preds = (preds > 0.5).astype(np.uint8) # Threshold = 0.35

        epoch_loss_val += loss.item()

        masks = masks.detach().cpu().numpy()

        dice_score+=calculate_dice_scores(masks, preds)*len(images)

    dice_score /= len(dataset_val)
    if dice_score > best_dice_score:
        best_dice_score = dice_score
        torch.save(model.state_dict(), model_dir+model_name+str(int(dice_score*100))+'.pt')
    scheduler.step(epoch_loss_val)
    print(f'Epoch {epoch+1}, train_loss: {epoch_loss/len(dataloader)} val_loss: {epoch_loss_val/len(dataloader_val)} dice_score: {dice_score}')
