#!/usr/bin/env python
# coding=utf-8

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import training
import wandb
from common import getters
from common.argparse import TrainingArguments
from common.utils import (increment_path, random_split, rle_encode,
                          seed_everything)
from dataset import transform
from dataset.dataset import SatelliteDataset
from model.unet import UNet
from training.trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)


def worker_init_fn(seed):
    import random
    import time

    import numpy as np
    seed = (seed + 1) * (int(time.time()) % 60)  # set random seed every epoch!
    random.seed(seed + 1)
    np.random.seed(seed)


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
    # seed_everything(cfg.seed)

    # Load model
    if cfg.architecture:  # Load from architecture
        logger.info("*** Loading from architecture ***")

        init_params = {
            "encoder_name": cfg.encoder_name, 
            "encoder_weights": cfg.encoder_weights, 
            "classes": cfg.classes, 
            "activation": cfg.activation
        }
        model = getters.get_model(architecture=cfg.architecture, init_params=init_params)    
    else:
        model = UNet()
    model.to(device)

    params = model.parameters()


    if cfg.do_train:
        logger.info("*** Train ***")

        cfg.output_dir = str(increment_path(
            path=Path(os.path.join(cfg.output_dir, cfg.architecture, cfg.encoder_name)) / "exp", 
            exist_ok=cfg.overwrite_output_dir, 
            mkdir=True
        ))


        # --------------------------------------------------
        # define datasets and dataloaders
        # --------------------------------------------------
        train_dataset = SatelliteDataset(
            data_dir=cfg.data_dir, 
            csv_file=cfg.train_file, 
            transform=transform.train_transform_2
        )
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=cfg.num_workers, 
            worker_init_fn=worker_init_fn
        )

        val_dataset = SatelliteDataset(
            data_dir=cfg.data_dir, 
            csv_file=cfg.train_file, 
            transform=transform.test_transform_1, 
            val=True
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=cfg.batch_size,
            shuffle=False, 
            num_workers=cfg.num_workers
        )

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")


        # --------------------------------------------------
        # define losses and metrics functions
        # --------------------------------------------------
        losses = {}
        #losses[cfg.losses] = getters.get_smp_loss(cfg.losses, init_params={'mode': 'binary'})
        losses[cfg.losses] = getters.get_loss(cfg.losses, init_params=None)

        metrics = {}
        metrics[cfg.metrics] = getters.get_metric(cfg.metrics, init_params=None)
        metrics["DiceScore"] = getters.get_metric("DiceScore", init_params=None)


        # --------------------------------------------------
        # define optimizer and scheduler
        # --------------------------------------------------
        optimizer = getters.get_optimizer(
            cfg.optimizer,
            model_params=params,
            init_params={"lr":cfg.lr},
        )

        if cfg.scheduler:
            scheduler = getters.get_scheduler(
                cfg.scheduler,
                optimizer,
                init_params={"epochs":cfg.epochs},
            )
        else:
            scheduler = None


        # --------------------------------------------------
        # define callbacks
        # --------------------------------------------------
        callbacks = []

        # add scheduler callback
        if scheduler is not None:
            callbacks.append(training.callbacks.Scheduler(scheduler))

        # add default logging and checkpoint callbacks
        if cfg.output_dir is not None:
            callbacks.append(training.callbacks.ModelCheckpoint(
                directory=cfg.output_dir,
                monitor='val_loss',
                save_best=True,
                save_last=True,
                save_top_k=0,
                mode="min",
                verbose=True,
            ))


        # --------------------------------------------------
        # start training
        # --------------------------------------------------
        print('Start training...')

        trainer = Trainer(model, model_device=device)
        trainer.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics,
        )

        trainer.train(
            train_dataloader=train_dataloader,
            valid_dataloader=val_dataloader,
            callbacks=callbacks,
            epochs=cfg.epochs,
            accumulation_steps=cfg.accumulation_steps,
            verbose=cfg.verbose,
        )

        if cfg.output_dir is not None:
            with open(os.path.join(cfg.output_dir, "config.json"), "w") as json_file:
                json.dump(cfg, json_file)


    if cfg.do_test:
        logger.info("*** Test ***")

        test_csv = os.path.join(cfg.data_dir, cfg.test_file)
        test_dataset = SatelliteDataset(data_dir=cfg.data_dir, csv_file=test_csv, transform=transform.test_transform_1, test=True)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        logger.info(f"Test dataset size: {len(test_dataset)}")
        
        model.load_state_dict(torch.load('..\\models\\effb1\\exp13\\checkpoints\\best.pth'))
        trainer = Trainer(model, model_device=device)
        trainer.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics,
        )

        mask = trainer.predict(
            dataloader=test_dataloader,
            verbose=cfg.verbose,
        )
        #test(model, test_dataloader, cfg.data_dir)




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
