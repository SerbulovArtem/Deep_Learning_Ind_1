from pathlib import Path
from typing import Sequence, Tuple, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset

class TestImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, extensions: Optional[Sequence[str]] = None):
        self.root = Path(root_dir)
        exts = extensions or (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.paths = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in exts])
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.root}")
        self.ids = [p.stem for p in self.paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.ids[idx]


def load_class_names(path: str = "data/classes.txt") -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]