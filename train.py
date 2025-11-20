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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
)


logger = logging.getLogger(__name__)


# Data transform

logger.info('Data Transformation')

transform = v2.Compose([
    v2.ToImage(),
    # v2.CenterCrop(224),
    # v2.Grayscale(),
    # v2.RandomPerspective(),
    # v2.RandomAffine(degrees=(30, 90), translate=(0.1, 0.2), scale=(0.5, 0.75)),
    # v2.RandomRotation(degrees=(0, 180)),
    # v2.ElasticTransform(alpha=200.0),
    # v2.ColorJitter(brightness=.5, hue=.3),
    # v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
    # v2.RandomInvert(),
    # v2.RandomPosterize(bits=2),
    # v2.RandomSolarize(threshold=117.0),
    # v2.RandomAdjustSharpness(sharpness_factor=2),
    # v2.RandomAutocontrast(),
    # v2.RandomEqualize(),
    # v2.AugMix(),
    # v2.RandomHorizontalFlip(p=0.5),
    # v2.RandomVerticalFlip(p=0.5),
    # v2.JPEG((5, 50)),
    v2.ToDtype(torch.float32, scale=True),
])

dataset = torchvision.datasets.ImageFolder("data/train/", transform=transform)


# Data Loaders

logger.info('Data Loading')

train_idx, val_idx = train_test_split(
    list(range(len(dataset))),
    test_size=0.2,
    stratify=dataset.targets,
    random_state=42
)

train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)

batch_size = 32
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=torch.cuda.is_available())

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

logger.info('Connecting to MLflow')

mlflow.autolog(disable=True)
mlflow.login()

from models import GoogLeNet

model = GoogLeNet(100).to(device)

logger.info(f"Creating model {model._get_name()}")

from trainer import Trainer
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)

trainer = Trainer(model=model, optimizer=optimizer)

logger.info("Sarting model training")

trainer.fit(
    train_loader=train_loader, 
    val_loader=val_loader, 
    epochs=5
)