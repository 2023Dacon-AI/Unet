#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import simple_parsing
import torch
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.argparse import TrainingArguments
from common.utils import (increment_path, random_split, rle_encode,
                          seed_everything, visualize, reverse_one_hot, colour_code_segmentation, crop_image)
from dataset.dataset import (ModifiedSatelliteDataset, SatelliteDataset,
                             transform, get_training_augmentation, get_validation_augmentation, get_preprocessing)
from model.unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
import random
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

import albumentations as album
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class_names = ['background', 'building']
# Get class RGB values
class_rgb_values = [[0, 0, 0], [255, 255, 255]]
select_classes = ['background', 'building']

# Get RGB values of required classes
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]


logger = logging.getLogger(__name__)


def train(model, train_epoch, valid_epoch, train_dataloader, val_dataloader, loss, optimizer, num_epochs, output_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    best_vloss = float('inf')

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for epoch in range(num_epochs):

        print('\nEpoch: {}'.format(epoch+1))
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(val_dataloader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            model_path = os.path.join(output_dir, 'UNet_{}_{}.pt'.format(timestamp, epoch+1))
            torch.save(model.state_dict(), model_path)
            print('Model saved!')


def main(args=None) -> None:
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        cfg: TrainingArguments = simple_parsing.parse(config_class=TrainingArguments, args="--config_path " + sys.argv[1], add_config_path_arg=True)
    else:
        cfg: TrainingArguments = simple_parsing.parse(TrainingArguments)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    logger.info(f"Configuration {cfg}")

    seed_everything(cfg.seed)
    cfg.output_dir = str(increment_path(Path(cfg.output_dir) / "exp", exist_ok=cfg.overwrite_output_dir, mkdir=True))


    train_csv = os.path.join(cfg.data_dir, cfg.train_file)
    test_csv = os.path.join(cfg.data_dir, cfg.test_file)
    

    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = class_names
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    ).to(device)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    total_dataset = ModifiedSatelliteDataset(data_dir=cfg.data_dir, augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), csv_file=train_csv)
    train_dataset, val_dataset = random_split(total_dataset, [0.9, 0.1])
        
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)


    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    
    if cfg.do_train:
        logger.info("*** Train ***")

        loss = DiceLoss()

        metrics = [
            IoU(threshold=0.5),
        ]

        # define optimizer
        optimizer = torch.optim.Adam([ 
            dict(params=model.parameters(), lr=0.0001),
        ])

        # define learning rate scheduler (not used in this NB)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=2, eta_min=5e-5,
        )
        train_epoch = TrainEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            device=device,
            verbose=True,
        )

        valid_epoch = ValidEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            device=device,
            verbose=True,
        )

        train(model, train_epoch, valid_epoch, train_dataloader, val_dataloader, loss, optimizer, cfg.num_epochs, cfg.output_dir)


    if cfg.do_test:

        logger.info("*** Test ***")

        model.load_state_dict(torch.load('../models\\exp37\\UNet_20230710_180428_39.pt', map_location=device))
        
        test_dataset = ModifiedSatelliteDataset(data_dir=cfg.data_dir, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), csv_file=test_csv, infer=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
        test_dataset_vis = ModifiedSatelliteDataset(data_dir=cfg.data_dir, augmentation=get_validation_augmentation(), csv_file=test_csv, infer=True)



        model.eval()
        with torch.no_grad():
            result = []
            for idx, images in enumerate(tqdm(test_dataloader)):
                image_vis = test_dataset_vis[idx].astype('uint8')
                images = images.float().to(device)
                outputs = model(images)
                outputs = torch.argmax(outputs, dim=1)  # Take the index of the maximum value along the channel dimension
                masks = outputs.detach().cpu().numpy()
                masks = masks.astype('uint8')
                masks = masks.clip(0, 1)

                vis_mask = colour_code_segmentation(masks, select_class_rgb_values)

                # visualize(
                #     original_image = image_vis,
                #     predicted_mask = vis_mask,
                # )

                for i in range(len(images)):
                    mask_rle = rle_encode(masks[i])
                    if len(mask_rle.split()) < 10: # 예측된 건물 픽셀이 아예 없는 경우 -1
                        result.append(-1)
                    else:
                        result.append(mask_rle)
            submit = pd.read_csv('../data/sample_submission.csv')
            submit['mask_rle'] = result
            submit.to_csv('./submit.csv', index=False)


if __name__ == "__main__":
    main()
