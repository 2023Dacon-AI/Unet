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

# Model
Encoder = 'resnet34'
Weights = 'imagenet'
prep_fun = smp.encoders.get_preprocessing_fn(
    Encoder,
    Weights
)

model = smp.Unet(
    encoder_name = Encoder,
    encoder_weights = Weights,
    in_channels = 3,
    classes = 1,
    activation = None
)
model = model.to(device)
model_name=''
model = model.load_state_dict(torch.load(model_name))

criterion = MixedLoss(alpha = 10.0,
                      gamma = 2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda epoch: 0.9 ** epoch,
    last_epoch=-1,
    verbose=False
)

#Transform
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),  # 좌우 대칭
        A.VerticalFlip(p=0.5),    # 상하 대칭
        A.RandomCrop(224,224),
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)
transform_val = A.Compose(
    [
        A.HorizontalFlip(p=0.5),  # 좌우 대칭
        A.VerticalFlip(p=0.5),    # 상하 대칭
        A.RandomCrop(224,224),
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)
transform_test = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)

batch_size= 32
epochs=50

dataset = SatelliteDataset(csv_file='./train_edited.csv', transform=transform, val=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

dataset_val = SatelliteDataset(csv_file='./train.csv', transform=transform_val, val=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

dataset_test = SatelliteDataset(csv_file='./test.csv', transform=transform_test, test=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

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

    scheduler.step()
    print("lr: ", optimizer.param_groups[0]['lr'])

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

    print(f'Epoch {epoch+1}, train_loss: {epoch_loss/len(dataloader)} val_loss: {epoch_loss_val/len(dataloader_val)} dice_score: {dice_score/len(dataset_val)}')