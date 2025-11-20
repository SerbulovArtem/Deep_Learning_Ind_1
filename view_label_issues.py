import os
import csv
import random
import argparse
import logging
from typing import List, Dict

import matplotlib.pyplot as plt
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def read_issues(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def pick_random_image_of_class(train_root: str, class_name: str, exclude_path: str | None = None) -> str:
    class_dir = os.path.join(train_root, class_name)
    if not os.path.isdir(class_dir):
        return ""
    files = [x for x in os.listdir(class_dir) if x.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        return ""
    # Build full paths and optionally exclude the issue image itself
    paths = [os.path.join(class_dir, x) for x in files]
    if exclude_path:
        excl_base = os.path.basename(exclude_path)
        paths = [p for p in paths if os.path.basename(p) != excl_base]
    if not paths:
        return ""
    return random.choice(paths)

def show_pair(issue_row: Dict, train_root: str, output_dir: str = None, show: bool = True):
    issue_path = issue_row["filepath"]
    given_class = issue_row["class_name"]
    predicted_class = issue_row["predicted_class_name"]

    # Random samples to compare: one from the given label, one from the predicted label
    given_sample_path = pick_random_image_of_class(train_root, given_class, exclude_path=issue_path)
    predicted_sample_path = pick_random_image_of_class(train_root, predicted_class)

    if not os.path.isfile(issue_path):
        logger.warning(f"Missing issue image: {issue_path}")
        return
    if not predicted_sample_path or not os.path.isfile(predicted_sample_path):
        logger.warning(f"No sample found for predicted class {predicted_class}")
        return
    if not given_sample_path or not os.path.isfile(given_sample_path):
        logger.warning(f"No sample found for given class {given_class} (excluding the issue image)")
        return

    # Build a 3-panel comparison: issue image, given-class sample, predicted-class sample
    panels = [
        (issue_path, f"Issue: Given {given_class} → Pred {predicted_class}"),
        (given_sample_path, f"Given sample: {given_class}"),
        (predicted_sample_path, f"Predicted sample: {predicted_class}"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4*len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (path, title) in zip(axes, panels):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to open {path}: {e}")
            return
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # CSV fields are loaded as strings; cast before numeric formatting
    try:
        pred_prob_val = float(issue_row['predicted_prob'])
    except (ValueError, TypeError):
        pred_prob_val = float('nan')
    try:
        self_conf_val = float(issue_row['self_confidence_given_label'])
    except (ValueError, TypeError):
        self_conf_val = float('nan')

    fig.suptitle(
        f"Index {issue_row['index']}  prob_pred={pred_prob_val:.3f}  self_conf={self_conf_val:.3e}",
        fontsize=10
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"issue_{issue_row['index']}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize label issues with suggested (predicted) class examples.")
    parser.add_argument("--csv", default="data/label_issues.csv", help="Path to label issues CSV.")
    parser.add_argument("--train-root", default="data/train", help="Root folder of training images.")
    parser.add_argument("--limit", type=int, default=10, help="Max number of issues to visualize.")
    parser.add_argument("--indices", type=str, default="", help="Comma-separated list of specific indices to show (overrides limit).")
    parser.add_argument("--save-dir", type=str, default="", help="If set, saves figures instead of only showing them.")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive windows; only save.")
    args = parser.parse_args()

    rows = read_issues(args.csv)
    logger.info(f"Loaded {len(rows)} issue rows")

    if args.indices:
        wanted = set(args.indices.split(","))
        rows = [r for r in rows if r["index"] in wanted]
        logger.info(f"Filtered to {len(rows)} specified indices")
    else:
        rows = rows[:args.limit]

    if not rows:
        logger.info("No rows to display")
        return

    for r in rows:
        logger.info(
            f"Index={r['index']} given={r['class_name']} predicted={r['predicted_class_name']} "
            f"pred_prob={float(r['predicted_prob']):.4f} self_conf={float(r['self_confidence_given_label']):.2e}"
        )
        show_pair(
            r,
            train_root=args.train_root,
            output_dir=args.save_dir or None,
            show=not args.no_show and not args.save_dir
        )

if __name__ == "__main__":
    main()