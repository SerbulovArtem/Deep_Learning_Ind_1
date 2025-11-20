import numpy as np
from cleanlab.filter import find_label_issues
import logging
import torch
import timm
from timm.data import resolve_model_data_config, create_transform
from models import ViT
import torchvision
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
)

logger.info("Loading saved model from models/VIT_model.pth for Cleanlab analysis")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
MODEL_NAME = "vit_base_patch16_224.augreg_in21k_ft_in1k"

loaded_base_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=100)
data_config = resolve_model_data_config(loaded_base_model)
loaded_model = ViT(loaded_base_model).to(device)
state_dict = torch.load("models/VIT_model.pth", map_location=device)

# Handle trainer save format
if isinstance(state_dict, dict) and "model_state" in state_dict:
    logger.info(f"Loaded container keys: {list(state_dict.keys())}")
    state_dict = state_dict["model_state"]

missing, unexpected = loaded_model.load_state_dict(state_dict, strict=False)
if missing:
    logger.warning(f"Missing keys after load (likely fine if loss/aux not needed): {missing}")
if unexpected:
    logger.warning(f"Unexpected keys ignored: {unexpected}")
loaded_model.eval()


val_transform   = create_transform(**data_config, is_training=False)

# Use full (non-augmented) dataset for consistent probabilities
full_eval_dataset = torchvision.datasets.ImageFolder("data/train/", transform=val_transform)
eval_loader_full = DataLoader(
    full_eval_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=1,
    pin_memory=torch.cuda.is_available()
)

num_classes = 100
pred_probs = np.zeros((len(full_eval_dataset), num_classes), dtype=np.float32)

with torch.inference_mode():
    idx = 0
    for imgs, _ in eval_loader_full:
        imgs = imgs.to(device)
        logits = loaded_model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        bsz = probs.shape[0]
        pred_probs[idx:idx+bsz] = probs
        idx += bsz

labels = np.array(full_eval_dataset.targets)

issue_indices = find_label_issues(
    labels=labels,
    pred_probs=pred_probs,
    return_indices_ranked_by="self_confidence"
)

logger.info(f"Potential label issues found: {len(issue_indices)}")

# Derive suggested labels from model predictions
predicted_labels = pred_probs.argmax(axis=1)
predicted_probs = pred_probs.max(axis=1)
given_probs = pred_probs[np.arange(len(full_eval_dataset)), labels]  # self-confidence

# Save issues with suggestions
import csv
issues_csv = "data/label_issues.csv"
with open(issues_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "index",
        "filepath",
        "given_label",
        "class_name",
        "predicted_label_id",
        "predicted_class_name",
        "predicted_prob",
        "self_confidence_given_label"
    ])
    for i in issue_indices:
        path, given_label = full_eval_dataset.samples[i]
        pred_lbl = int(predicted_labels[i])
        writer.writerow([
            i,
            path,
            int(given_label),
            full_eval_dataset.classes[given_label],
            pred_lbl,
            full_eval_dataset.classes[pred_lbl],
            float(predicted_probs[i]),
            float(given_probs[i]),
        ])

logger.info(f"Label issues written to {issues_csv}")