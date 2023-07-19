#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import wandb

import numpy as np
import pandas as pd
import simple_parsing
import torch
from common.argparse import TrainingArguments
from common.utils import (increment_path, random_split, rle_encode,
                          seed_everything)
from dataset.dataset import SatelliteDataset, transform
from model.unet import UNet
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)


def train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, output_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = float('inf')

    for epoch in range(num_epochs):
        model.train(True)
        train_loss = 0.
        val_loss = 0.

        for data in tqdm(train_dataloader):
            images = data[0].float().to(device)             # (batch_size, 3, 224, 224)
            masks =  data[1].float().to(device)             # (batch_size, 224, 224)

            optimizer.zero_grad()
            outputs = model(images)                         # (batch_size, 1, 224, 224)
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Evaluate on the validation set
        model.eval()
        with torch.no_grad():
            for vdata in tqdm(val_dataloader):
                vimages, vmasks = vdata[0].float().to(device), vdata[1].float().to(device)
                voutputs = model(vimages)
                vloss = criterion(voutputs, vmasks.unsqueeze(1))
                val_loss += vloss

        val_loss /= len(val_dataloader)

        print('EPOCH {}: Train Loss {}, Val Loss {}'.format(epoch + 1, train_loss, val_loss))
        wandb.log({'Train Loss': train_loss, 'epoch': epoch+1})
        wandb.log({'Val Loss': val_loss, 'epoch': epoch+1})

        if val_loss < best_vloss:
            best_vloss = val_loss
            model_path = os.path.join(output_dir, 'UNet_{}_{}.pt'.format(timestamp, epoch+1))
            torch.save(model.state_dict(), model_path)


def test(model, test_dataloader, data_dir):
        model.eval()
        with torch.no_grad():
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
            submit = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
            submit['mask_rle'] = result
            submit.to_csv(os.path.join(data_dir, 'submit.csv'), index=False)


def main(cfg=None) -> None:
    seed_everything(cfg.seed)

    # Load model
    if cfg.model_path:  # Load from checkpoint
        model = UNet()
        model.load_state_dict(torch.load(cfg.model_path))
        model.to(device)
    else:
        model = UNet().to(device)
    

    if cfg.do_train:
        logger.info("*** Train ***")

        cfg.output_dir = str(increment_path(Path(cfg.output_dir) / "exp", exist_ok=cfg.overwrite_output_dir, mkdir=True))
        train_csv = os.path.join(cfg.data_dir, cfg.train_file)
        
        train_dataset = SatelliteDataset(data_dir=cfg.data_dir, csv_file=train_csv, transform=transform, val=False)
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

        val_dataset = SatelliteDataset(data_dir=cfg.data_dir, csv_file=train_csv, transform=transform, val=True)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")


        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        train(model, train_dataloader, val_dataloader, criterion, optimizer, cfg.num_epochs, cfg.output_dir)


    if cfg.do_test:
        logger.info("*** Test ***")

        test_csv = os.path.join(cfg.data_dir, cfg.test_file)
        test_dataset = SatelliteDataset(data_dir=cfg.data_dir, csv_file=test_csv, transform=transform, infer=True)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        logger.info(f"Test dataset size: {len(test_dataset)}")

        test(model, test_dataloader, cfg.data_dir)




if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        cfg: TrainingArguments = simple_parsing.parse(config_class=TrainingArguments, args="--config_path " + sys.argv[1], add_config_path_arg=True)
        wandb.init(
            project="dacon_satellite_segmentation",
            config=cfg
        )
    else:
        cfg: TrainingArguments = simple_parsing.parse(TrainingArguments)

    main(cfg)
