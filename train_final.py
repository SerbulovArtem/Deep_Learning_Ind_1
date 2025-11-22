# train_final.py
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

import timm
from timm.data import resolve_model_data_config, create_transform
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import mlflow
from optuna_trainer import Trainer
from models import ViT, ConvNext

from dotenv import load_dotenv

load_dotenv()

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

SPLITS_PATH = Path("data/splits/splits_80_10_10.json")
HPO_BEST_PATH = Path("data/hpo/best_convnext.json")
DATA_ROOT = Path("data/train")


def build_datasets_90_10(model_name: str):
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    base = timm.create_model(model_name, pretrained=True, num_classes=100)
    cfg = resolve_model_data_config(base)
    tf_train = create_transform(**cfg, is_training=True)
    tf_eval = create_transform(**cfg, is_training=False)

    ds_train_all = datasets.ImageFolder(str(DATA_ROOT), transform=tf_train)
    ds_eval_all = datasets.ImageFolder(str(DATA_ROOT), transform=tf_eval)

    # 90% = train + val; 10% = test
    train_plus_val_idx = splits["train_idx"] + splits["val_idx"]
    test_idx = splits["test_idx"]

    train_plus_val_ds = Subset(ds_train_all, train_plus_val_idx)
    test_ds = Subset(ds_eval_all, test_idx)

    return train_plus_val_ds, test_ds


def build_model(model_name: str, device: str):
    base = timm.create_model(model_name, pretrained=True, num_classes=100)
    if "vit" in model_name:
        model = ViT(base).to(device)
    elif "convnext" in model_name:
        model = ConvNext(base).to(device)
    else:
        model = base.to(device)
    return model


def main():
    with open(HPO_BEST_PATH) as f:
        cfg = json.load(f)
    model_name = cfg["model_name"]
    params = cfg["best_params"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, test_ds = build_datasets_90_10(model_name)

    batch_size = params["batch_size"]
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    t_max = params["T_max"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=1, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=1, pin_memory=torch.cuda.is_available())

    model = build_model(model_name, device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max)

    mlflow.autolog(disable=True)
    mlflow.login()
    exp = mlflow.set_experiment(
        experiment_name=f"/Users/artemserbulov117@gmail.com/100 Butterflies FINAL {model._get_name()}"
    )

    from mlflow import pytorch as mlflow_pytorch

    with mlflow.start_run(experiment_id=exp.experiment_id):
        mlflow.log_params(
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "T_max": t_max,
            }
        )

        trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler)
        # here we can treat test_loader as "val" loader just for logging
        trainer.fit(train_loader=train_loader, val_loader=test_loader, epochs=10)

        # save final model and log to MLflow
        save_path = f"models/{model._get_name()}_final.pth"
        trainer.save(save_path, include_optimizer=False, include_scheduler=False)
        mlflow.log_artifact(save_path, artifact_path="checkpoints")

        mlflow_pytorch.log_model(model, artifact_path="model")

        logger.info(f"Final model saved at: {save_path}")


if __name__ == "__main__":
    main()