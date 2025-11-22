import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from scipy.cluster.hierarchy import linkage, leaves_list
from torchvision import datasets
from timm.data import resolve_model_data_config, create_transform
from models import ViT, ConvNext

MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"
MODEL_FILE = "models/ConvNext_final.pth"
DATA_DIR = "data/train"
BATCH = 32
OUT_DIR = "data/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

tmp = timm.create_model(MODEL_NAME, pretrained=False, num_classes=100)
cfg = resolve_model_data_config(tmp)
val_tf = create_transform(**cfg, is_training=False)

ds = datasets.ImageFolder(DATA_DIR, transform=val_tf)
loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=1)

if "vit" in MODEL_NAME:
    base = timm.create_model(MODEL_NAME, pretrained=False, num_classes=100)
    model = ViT(base)
elif "convnext" in MODEL_NAME:
    base = timm.create_model(MODEL_NAME, pretrained=False, num_classes=100)
    model = ConvNext(base)
else:
    raise ValueError("Unsupported model")

state = torch.load(MODEL_FILE, map_location=device)

if isinstance(state, dict) and "model_state" in state:
    ckpt = state
    model_state = ckpt["model_state"]
    missing, unexpected = model.load_state_dict(model_state, strict=True)
else:
    missing, unexpected = model.load_state_dict(state, strict=False)

model.to(device).eval()

preds, targets = [], []
with torch.inference_mode():
    for x,y in loader:
        x = x.to(device)
        p = model(x).argmax(1).cpu().numpy()
        preds.append(p); targets.append(y.numpy())
preds = np.concatenate(preds)
targets = np.concatenate(targets)

cm = confusion_matrix(targets, preds)
cm_row = cm / (cm.sum(1, keepdims=True)+1e-12)

classes = ds.classes

# Per-class metrics
prec, rec, f1, support = precision_recall_fscore_support(targets, preds, labels=range(len(classes)), zero_division=0)
metrics = np.stack([support, prec, rec, f1], axis=1)
np.savetxt(f"{OUT_DIR}/per_class_metrics.csv",
           np.column_stack([np.array(classes), metrics]),
           fmt="%s", delimiter=",")

# Top confusing pairs (exclude diagonal)
off = cm.copy()
np.fill_diagonal(off, 0)
pairs = []
for i in range(off.shape[0]):
    for j in range(off.shape[1]):
        if off[i,j] > 0:
            pairs.append((off[i,j], i, j))
pairs.sort(reverse=True)
top_k = pairs[:25]
with open(f"{OUT_DIR}/top_confusions.txt","w") as f:
    for cnt,i,j in top_k:
        f.write(f"{classes[i]} -> {classes[j]}: {cnt}\n")

# Plot per-class recall sorted
order_rec = np.argsort(rec)
plt.figure(figsize=(10,6))
plt.bar(range(len(classes)), rec[order_rec])
plt.xticks(range(len(classes)), [classes[i] for i in order_rec], rotation=90, fontsize=6)
plt.ylabel("Recall")
plt.title("Per-class Recall (sorted)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/per_class_recall.png", dpi=180)
plt.close()

# Reduced confusion matrix (worst 20 recall)
worst20 = order_rec[:20]
sub = cm_row[worst20][:, worst20]
plt.figure(figsize=(8,6))
plt.imshow(sub, cmap="viridis", interpolation="nearest")
plt.colorbar()
plt.xticks(range(20), [classes[i] for i in worst20], rotation=90, fontsize=7)
plt.yticks(range(20), [classes[i] for i in worst20], fontsize=7)
plt.title("Confusion (row-normalized) - Worst 20 Recall Classes")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_worst20.png", dpi=180)
plt.close()

# Clustered confusion matrix
# Distance = 1 - Jaccard-like over confusion profiles
profiles = cm_row
Z = linkage(profiles, method="average", metric="euclidean")
leaf_order = leaves_list(Z)
cluster_cm = cm_row[leaf_order][:, leaf_order]
plt.figure(figsize=(10,8))
plt.imshow(cluster_cm, cmap="viridis", interpolation="nearest")
plt.colorbar()
plt.xticks([], [])
plt.yticks([], [])
plt.title("Clustered Confusion Matrix (row-normalized)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_clustered.png", dpi=180)
plt.close()

print("Artifacts written to", OUT_DIR)

param_sum = sum(p.abs().sum().item() for p in model.parameters())
print("Param L1 sum:", param_sum)
acc = (preds == targets).mean()
print("Local accuracy:", acc)