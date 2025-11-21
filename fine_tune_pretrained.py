import os
import sys
from dotenv import load_dotenv

load_dotenv()

import torch
import mlflow
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import logging


import timm
from timm.data import resolve_model_data_config, create_transform
from torch import nn


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

logger.info('Data Transformation')

MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"  # or: "deit_small_patch16_224.fb_in1k", "eva02_base_patch14_224.mim_m38m_ft_in22k_in1k", "convnext_base.fb_in22k_ft_in1k"

# MODEL_NAME = "vit_base_patch16_224.augreg_in21k_ft_in1k"


_tmp_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=100)
data_config = resolve_model_data_config(_tmp_model)

train_transform = create_transform(**data_config, is_training=True)
val_transform   = create_transform(**data_config, is_training=False)

from torchvision import datasets
dataset_train = datasets.ImageFolder("data/train/", transform=train_transform)
dataset_val   = datasets.ImageFolder("data/train/", transform=val_transform)

# Data Loaders
logger.info('Data Loading')

train_idx, val_idx = train_test_split(
    list(range(len(dataset_train))),
    test_size=0.2,
    stratify=dataset_train.targets,
    random_state=42
)

train_ds = Subset(dataset_train, train_idx)
val_ds   = Subset(dataset_val,   val_idx)

batch_size = 32
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=torch.cuda.is_available())

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

logger.info('Connecting to MLflow')

mlflow.autolog(disable=True)
mlflow.login()

from models import ViT, ConvNext

base_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=100)

if "vit" in MODEL_NAME:
    model = ViT(base_model).to(device)
elif "convnext" in MODEL_NAME:
    model = ConvNext(base_model).to(device)

logger.info(f"Creating model {model._get_name()}")

from trainer import Trainer
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=15)

trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler)

logger.info("Sarting model training")

trainer.fit(
    train_loader=train_loader, 
    val_loader=val_loader, 
    epochs=10
)

trainer.save(path=f"models/{model._get_name()}_model.pth")

from dataset import TestImageDataset, load_class_names
from torch.utils.data import DataLoader

test_ds = TestImageDataset("data/test", transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=1, pin_memory=torch.cuda.is_available())

class_names = load_class_names("data/classes.txt")
trainer.create_submission(test_loader, class_names, f"data/submission_{model._get_name()}.csv")