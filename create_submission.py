import os
from dotenv import load_dotenv

load_dotenv()

import torch
import timm
from timm.data import resolve_model_data_config, create_transform
from torch.utils.data import DataLoader

from dataset import TestImageDataset, load_class_names
from trainer import Trainer
from models import ViT, ConvNext


MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"  # adjust if needed
MODEL_CHECKPOINT = "models/ConvNext_final.pth"


def load_model_and_trainer(checkpoint_path: str) -> Trainer:
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    base_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=100)
    if "vit" in MODEL_NAME:
        model = ViT(base_model).to(device)
    elif "convnext" in MODEL_NAME:
        model = ConvNext(base_model).to(device)
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")

    trainer = Trainer(model=model, optimizer=None, scheduler=None)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)
    trainer.model.load_state_dict(state["model_state"])

    return trainer


def main():
    model_path = MODEL_CHECKPOINT
    trainer = load_model_and_trainer(model_path)

    tmp_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=100)
    data_config = resolve_model_data_config(tmp_model)
    val_transform = create_transform(**data_config, is_training=False)

    test_ds = TestImageDataset("data/test", transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=1, pin_memory=torch.cuda.is_available())

    class_names = load_class_names("data/classes.txt")

    submission_path = f"data/submission_{trainer.model._get_name()}.csv"
    trainer.create_submission(test_loader, class_names, submission_path)


if __name__ == "__main__":
    main()
