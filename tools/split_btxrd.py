# ---------------------------------------------------------------
# Split the full BTXRD release into train / val / test.
#
# Input: the original flat layout
#   <src>/images/*.jpeg|*.jpg        all radiographs (with and without tumor)
#   <src>/Annotations/*.json         LabelMe files, only for tumor images
#   <src>/dataset.xlsx               per-image metadata (tumor, benign, malignant, ...)
#
# Every image gets one image-level type:
#   normal    -> tumor == 0 (no annotation, no mask)
#   benign    -> benign == 1
#   malignant -> malignant == 1
# The split is stratified by this type, so normal / benign / malignant keep the
# same proportions in every subset.
#
# Output (with --dst): the layout read by datasets/btxrd.py
#   <dst>/split.csv                  image_id,split,type
#   <dst>/{train,val,test}/images/
#   <dst>/{train,val,test}/Annotations/   (only tumor images have a json)
#
# Usage:
#   python tools/split_btxrd.py --src /path/to/BTXRD --dst /path/to/BTXRD_split
#   python tools/split_btxrd.py --src /path/to/BTXRD --csv split.csv   # csv only
# ---------------------------------------------------------------

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
IMAGE_TYPES = ("normal", "benign", "malignant")
IMG_SUFFIXES = (".jpeg", ".jpg", ".png")


def read_image_types(src: Path) -> dict[str, str]:
    """image stem -> normal | benign | malignant, from dataset.xlsx."""
    import pandas as pd

    df = pd.read_excel(src / "dataset.xlsx")
    types = {}
    for _, row in df.iterrows():
        stem = Path(str(row["image_id"])).stem
        if int(row["tumor"]) == 0:
            types[stem] = "normal"
        elif int(row["malignant"]) == 1:
            types[stem] = "malignant"
        elif int(row["benign"]) == 1:
            types[stem] = "benign"
    return types


def make_split(
    src,
    ratios: tuple[float, float, float] = (0.80, 0.15, 0.05),
    seed: int = 42,
) -> list[dict]:
    """Return rows {image_id, split, type} for every usable image in <src>/images."""
    src = Path(src)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    types = read_image_types(src)
    ann_dir = src / "Annotations"

    by_type = defaultdict(list)
    skipped = Counter()
    for img in sorted((src / "images").iterdir()):
        if img.suffix.lower() not in IMG_SUFFIXES:
            continue
        image_type = types.get(img.stem)
        if image_type is None:
            skipped["not in dataset.xlsx"] += 1
            continue
        if image_type != "normal" and not (ann_dir / f"{img.stem}.json").exists():
            skipped["tumor without annotation"] += 1
            continue
        by_type[image_type].append(img.name)

    if skipped:
        print(f"Skipped: {dict(skipped)}")

    rng = random.Random(seed)
    rows = []
    for image_type in IMAGE_TYPES:
        names = by_type[image_type]
        rng.shuffle(names)
        n_val = round(len(names) * ratios[1])
        n_test = round(len(names) * ratios[2])
        n_train = len(names) - n_val - n_test
        for i, name in enumerate(names):
            split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            rows.append({"image_id": name, "split": split, "type": image_type})

    rows.sort(key=lambda r: r["image_id"])
    return rows


def write_csv(rows: list[dict], path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "split", "type"])
        writer.writeheader()
        writer.writerows(rows)


def copy_split(src, dst, rows: list[dict]) -> None:
    src, dst = Path(src), Path(dst)
    for split in SPLITS:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "Annotations").mkdir(parents=True, exist_ok=True)

    for row in rows:
        img = src / "images" / row["image_id"]
        shutil.copy2(img, dst / row["split"] / "images" / img.name)
        ann = src / "Annotations" / f"{img.stem}.json"
        if row["type"] != "normal" and ann.exists():
            shutil.copy2(ann, dst / row["split"] / "Annotations" / ann.name)

    write_csv(rows, dst / "split.csv")
    if (src / "dataset.xlsx").exists():
        shutil.copy2(src / "dataset.xlsx", dst / "dataset.xlsx")


def summary(rows: list[dict]) -> str:
    counts = Counter((r["split"], r["type"]) for r in rows)
    lines = [f"{'':6s}" + "".join(f"{t:>11s}" for t in IMAGE_TYPES) + f"{'total':>9s}"]
    for split in SPLITS:
        n = [counts[(split, t)] for t in IMAGE_TYPES]
        lines.append(f"{split:6s}" + "".join(f"{x:11d}" for x in n) + f"{sum(n):9d}")
    lines.append(f"{'total':6s}" + "".join(
        f"{sum(counts[(s, t)] for s in SPLITS):11d}" for t in IMAGE_TYPES
    ) + f"{len(rows):9d}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="folder with images/, Annotations/, dataset.xlsx")
    parser.add_argument("--dst", help="output folder (train/val/test sub-folders + split.csv)")
    parser.add_argument("--csv", help="only write split.csv to this path")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.80, 0.15, 0.05))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dst and not args.csv:
        parser.error("give --dst and/or --csv")
    if args.dst and Path(args.dst).exists() and any(Path(args.dst).iterdir()):
        parser.error(f"{args.dst} already exists and is not empty")

    rows = make_split(args.src, tuple(args.ratios), args.seed)
    if args.csv:
        write_csv(rows, args.csv)
    if args.dst:
        copy_split(args.src, args.dst, rows)
    print(summary(rows))


if __name__ == "__main__":
    main()
