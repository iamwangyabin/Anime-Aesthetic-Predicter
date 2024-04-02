import os
import argparse
import numpy as np
import pandas as pd
import wandb
import clip
import hydra
from datetime import datetime
from omegaconf import OmegaConf, ListConfig
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


import pytorch_lightning as pl
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint



class Trainer(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = OP_ENABLE_KTLS
        self.save_hyperparameters()
        self.backbone, _ = clip.load("ViT-L/14", device="cpu")

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.header = nn.Sequential(
            nn.Linear(768, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )


    def training_step(self, batch):
        x, y = batch
        features = self.backbone.encode_image(x) 
        scores = self.header(features)
        loss = F.mse_loss(scores, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        features = self.backbone.encode_image(x) 
        scores = self.header(features)
        loss = F.mse_loss(scores, y)
        self.log("val_loss", loss)
        return loss


    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.opt.train.optimizer(optparams)
        scheduler = self.opt.train.scheduler(optimizer)
        return [optimizer], [scheduler]



class ScoredImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = {
            'worst': 0,
            'low': 1,
            'normal': 2,
            'good': 3,
            'great': 4,
            'best': 5,
            'masterpiece': 6
        }
        self.data = []
        for category, label in self.labels.items():
            category_dir = os.path.join(self.root_dir, category)
            for img_file in os.listdir(category_dir):
                if img_file.lower().endswith(('.webp')):
                    self.data.append((os.path.join(category_dir, img_file), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def set_transform(self, transform):
        self.transform = transform




def load_config_with_cli(path, args_list=None, remove_undefined=True):
    cfg = OmegaConf.load(path)
    cfg_cli = OmegaConf.from_cli(args_list)
    cfg = OmegaConf.merge(cfg, cfg_cli)
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)
    wandb.login(key = 'a4d3a740e939973b02ac59fbd8ed0d6a151df34b')

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                            std=[0.26862954, 0.26130258, 0.27577711])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
    ])


    dataset = ScoredImageDataset(root_dir=conf.dataset.dataroot)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataset.dataset.set_transform(train_transforms)
    val_dataset.dataset.set_transform(val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=conf.dataset.batch_size, shuffle=True, num_workers=conf.dataset.loader_workers)
    val_loader = DataLoader(val_dataset, batch_size=conf.dataset.batch_size, shuffle=False, num_workers=conf.dataset.loader_workers)

    today_str = conf.name +"_"+ datetime.now().strftime('%Y%m%d_%H_%M_%S')

    wandb_logger = WandbLogger(name=today_str, project='DeepfakeDetection',
                               job_type='train', group=conf.name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join('logs', today_str),
        filename='{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )

    model = Trainer(conf)
    trainer = L.Trainer(logger=wandb_logger, max_epochs=conf.train.train_epochs, accelerator="gpu", devices=conf.train.gpu_ids,
                        callbacks=[checkpoint_callback],
                        val_check_interval=0.1,
                        precision="16")

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.save_checkpoint(os.path.join('logs', today_str, "last.ckpt"))












