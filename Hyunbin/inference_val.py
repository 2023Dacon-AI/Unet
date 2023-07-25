import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Dataset import SatelliteDataset
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
model.load_state_dict(torch.load(model_dir+model_name+'.pt'))


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

batch_size=64

dataset_val = SatelliteDataset(csv_file='./train.csv', transform=transform_val, val=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

with torch.no_grad():
    model.eval()
    dice_score=0
    for images, masks in tqdm(dataloader_val):
        images = images.float().to(device)
        masks = masks.float().to(device)

        outputs = model(images)

        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = np.squeeze(preds, axis=1)
        preds = (preds > 0.5).astype(np.uint8) # Threshold = 0.35

        masks = masks.detach().cpu().numpy()

        dice_score+=calculate_dice_scores(masks,preds) * len(images)
    dice_score /= dataset_val
    print(f"dice score for validation: {dice_score}")