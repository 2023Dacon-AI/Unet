import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from Dataset import SatelliteDataset
from Dataset import rle_encode
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_dir = '/content/drive/MyDrive/'
model_name='unet'

# Model
Encoder = 'regnet_y_128gf'
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
model = model.load_state_dict(torch.load(model_dir+model_name+'_best.pth'))
model = model.to(device)

transform_test = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)

batch_size=32

dataset_test = SatelliteDataset(csv_file='./test.csv', transform=transform_test, test=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(dataloader_test):
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
        submit.to_csv('/content/drive/MyDrive/submit.csv', index=False)