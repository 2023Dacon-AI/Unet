#!/usr/bin/env python
# coding=utf-8

import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import simple_parsing
import torch
from torch.utils.data import DataLoader

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


def main(cfg=None) -> None:
    wandb.init()
    cfg = TrainingArguments()
    for key, value in wandb.config.items():
        if value == "true":
            value = True
        elif value == "false":
            value = False
        setattr(cfg, key, value)


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

        state_dict = torch.load(os.path.join("..\\models\\pretrained\\stage1\\effb1-f0\\checkpoints", "best.pth"))["state_dict"]
        model.load_state_dict(state_dict)

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
            transform=getattr(transform, cfg.train_transform),
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
            transform=getattr(transform, cfg.test_transform),
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
            if cfg.scheduler == "PolyLR":
                init_params = {"epochs":cfg.epochs}
            else:
                init_params = None
            scheduler = getters.get_scheduler(
                cfg.scheduler,
                optimizer,
                init_params=init_params,
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
                config = dataclasses.asdict(cfg)
                config["wandb_url"] = wandb.run.get_url()
                json.dump(config, json_file, indent=4)


    if cfg.do_test:
        logger.info("*** Test ***")

        if cfg.do_train: # If was on train, load best model
            state_dict = torch.load(os.path.join(cfg.output_dir, "best.pth"))["state_dict"]
            model.load_state_dict(state_dict)
        else:
            cfg.output_dir = os.path.join(
            cfg.output_dir, 
            cfg.architecture, 
            cfg.encoder_name, 
            "exp"                      # Change experiment folder
            )
            state_dict = torch.load(os.path.join(cfg.output_dir, "best.pth"))["state_dict"]
            model.load_state_dict(state_dict)

        test_dataset = SatelliteDataset(
            data_dir=cfg.data_dir, 
            csv_file=cfg.test_file, 
            transform=transform.test_transform_1, 
            test=True
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=cfg.num_workers
        )
        logger.info(f"Test dataset size: {len(test_dataset)}")
        
        trainer = Trainer(model, model_device=device)

        results = trainer.predict(
            dataloader=test_dataloader,
            verbose=cfg.verbose,
        )
        submit = pd.read_csv(os.path.join(cfg.data_dir, 'sample_submission.csv'))
        submit['mask_rle'] = results
        submit.to_csv(os.path.join(cfg.output_dir, 'submit.csv'), index=False)



if __name__ == "__main__":

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        cfg: TrainingArguments = simple_parsing.parse(config_class=TrainingArguments, args="--config_path " + sys.argv[1], add_config_path_arg=True)

    else:
        cfg: TrainingArguments = simple_parsing.parse(TrainingArguments)

    # Define sweep config
    sweep_configuration = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'val_DiceScore'},
        'parameters': 
        {
            'overwrite_output_dir': {'value': "false"},
            'accumulation_steps': {'max': 8, 'min': 1},
            'encoder_weights': {'value': "imagenet"},
            'verbose': {'value': "true"},
            'encoder_name': {'value': "efficientnet-b1"},
            'architecture': {'value': "Unet"},
            'num_workers': {'value': 4},
            'train_file': {'value': "train_drop.csv"},
            'output_dir': {'value': "../models"},
            'batch_size': {'value': 20},
            'activation': {'value': "sigmoid"},
            'test_file': {'value': "test.csv"},
            'scheduler': {'values': ["PolyLR", "LinearLR"]},
            'optimizer': {'values': ["Adam", "AdamWarmup"]},
            'do_train': {'value': "true"},
            'data_dir': {'value': "../data"},
            'metrics': {'value': "MicroIoU"},
            'do_test': {'value': "true"},
            'classes': {'value': 1},
            'losses': {'values': ["DiceLoss","FocalDiceLoss"]},
            'epochs': {'value': 3},
            'train_transform': {'values': ['train_transform_2', 'train_transform_3']},
            'test_transform': {'value': "test_transform_1"},
            'seed': {'min':21, 'max': 84},
            'lr': {'max': 0.0002, 'min': 0.00005, 'distribution': 'uniform'},
        }
    }

    # Initialize sweep by passing in config. 

    sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='dacon_satellite_segmentation'
    )

    wandb.agent(sweep_id, function=main, count=10)
