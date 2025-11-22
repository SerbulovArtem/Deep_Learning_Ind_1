# optuna_search.py
import json
from pathlib import Path

import optuna
import torch
import mlflow
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

import timm
from timm.data import resolve_model_data_config, create_transform
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from optuna_trainer import Trainer
from models import ViT, ConvNext

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

SPLITS_PATH = Path("data/splits/splits_80_10_10.json")
DATA_ROOT = Path("data/train")

from dotenv import load_dotenv

load_dotenv()

def get_display_model_name(model_name: str) -> str:
    if "convnext" in model_name.lower():
        return "ConvNext"
    if "vit" in model_name.lower():
        return "ViT"
    return model_name  # fallback

def build_datasets(model_name: str):
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    base = timm.create_model(model_name, pretrained=True, num_classes=100)
    cfg = resolve_model_data_config(base)
    tf_train = create_transform(**cfg, is_training=True)
    tf_val = create_transform(**cfg, is_training=False)

    ds = datasets.ImageFolder(str(DATA_ROOT), transform=tf_train)
    ds_val = datasets.ImageFolder(str(DATA_ROOT), transform=tf_val)

    train_ds = Subset(ds, splits["train_idx"])
    val_ds = Subset(ds_val, splits["val_idx"])
    return train_ds, val_ds


def build_model(model_name: str, device: str):
    base = timm.create_model(model_name, pretrained=True, num_classes=100)
    if "vit" in model_name:
        model = ViT(base).to(device)
    elif "convnext" in model_name:
        model = ConvNext(base).to(device)
    else:
        model = base.to(device)
    return model


def objective(trial: optuna.Trial, model_name: str, train_ds, val_ds, device: str, trial_epochs: int):
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=1, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=1, pin_memory=torch.cuda.is_available())

    model = build_model(model_name, device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=trial_epochs)
    
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(
            {
                "trial_number": trial.number,
                "model_name": model_name,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "T_max": t_max,
            }
        )
        trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler)
        
        val_acc = trainer.fit(train_loader, val_loader, epochs=trial_epochs, use_mlflow=True)

    logger.info(f"Finished trial {trial.number} | val_acc={val_acc:.4f}")

    del model, optimizer, scheduler, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_acc


def main():
    model_name = "convnext_base.fb_in22k_ft_in1k"
    display_name = get_display_model_name(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Starting Optuna search for {display_name} on device={device}")

    train_ds, val_ds = build_datasets(model_name)
    
    mlflow.autolog(disable=True)
    mlflow.login()
    exp = mlflow.set_experiment(
        experiment_name=f"/Users/artemserbulov117@gmail.com/100 Butterflies {display_name}"
    )

    with mlflow.start_run(experiment_id=exp.experiment_id) as parent_run:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda t: objective(t, model_name, train_ds, val_ds, device, trial_epochs=5),
            n_trials=2,
        )

        best = {
            "model_name": model_name,
            "best_value": study.best_value,
            "best_params": study.best_params,
        }
        out_dir = Path("data/hpo")
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / "best_convnext.json", "w") as f:
            json.dump(best, f, indent=2)

        logger.info(f"Best val_acc={study.best_value}")
        logger.info(f"Best params={study.best_params}")
        logger.info(f"Optuna finished.")


if __name__ == "__main__":
    main()