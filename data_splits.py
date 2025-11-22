# data_splits.py
import json
from pathlib import Path

from torchvision import datasets
from sklearn.model_selection import train_test_split

DATA_ROOT = Path("data/train")
OUT_DIR = Path("data/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    ds = datasets.ImageFolder(str(DATA_ROOT))
    all_idx = list(range(len(ds)))

    train_idx, holdout_idx = train_test_split(
        all_idx,
        test_size=0.2,
        stratify=ds.targets,
        random_state=42,
    )

    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=0.5,
        stratify=[ds.targets[i] for i in holdout_idx],
        random_state=42,
    )

    splits = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }

    with open(OUT_DIR / "splits_80_10_10.json", "w") as f:
        json.dump(splits, f)

    print("Saved splits to", OUT_DIR / "splits_80_10_10.json")


if __name__ == "__main__":
    main()