# ---------------------------------------------------------------
# Prepare BTXRD for training: get the data, split it if needed, and write a
# resized copy (the folder given to --data.path).
#
# Source, one of:
#   --kaggle-dataset owner/name   downloaded with kagglehub (Kaggle credentials,
#                                 if needed, from KAGGLE_USERNAME / KAGGLE_KEY or
#                                 Colab Secrets with the same names)
#   --src PATH                    a folder or a .zip
# The source may be already split (split.csv + train/val/test, as written by
# tools/split_btxrd.py) or the raw flat release (images/, Annotations/,
# dataset.xlsx), which is then split 80:15:5 stratified by image type.
#
# Output: <out>/split.csv and <out>/{train,val,test}/{images,Annotations}/, where
# images are downscaled so that the longest side is <= --max-side (original
# radiographs are 2000-3000 px, far more than the 640 px the model sees) and
# polygons are rescaled accordingly. Normal images have no annotation.
#
# Usage:
#   python tools/prepare_btxrd.py --kaggle-dataset nguyetanhmai/btxrd-full-split --out /content/btxrd_cache
#   python tools/prepare_btxrd.py --src ../BTXRD_full_backup --out ../btxrd_cache
# ---------------------------------------------------------------

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from split_btxrd import IMG_SUFFIXES, SPLITS, make_split, summary, write_csv


def download_kaggle_dataset(handle: str) -> Path:
    if not os.environ.get("KAGGLE_KEY"):
        try:  # Colab Secrets
            from google.colab import userdata

            for name in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
                try:
                    os.environ[name] = userdata.get(name)
                except Exception:
                    pass
        except ImportError:
            pass

    import kagglehub

    print(f"Downloading Kaggle dataset {handle}")
    return Path(kagglehub.dataset_download(handle))


def find_root(base: Path, max_depth: int = 4) -> Path | None:
    """Folder with split.csv (already split), else folder with images/ + dataset.xlsx."""
    for name in ("split.csv", "dataset.xlsx"):
        for depth in range(max_depth + 1):
            for hit in sorted(base.glob("*/" * depth + name)):
                if name == "split.csv" or (hit.parent / "images").is_dir():
                    return hit.parent
    return None


def resolve_source(args) -> Path:
    if args.kaggle_dataset:
        base = download_kaggle_dataset(args.kaggle_dataset)
    else:
        base = Path(args.src)
    if base.suffix.lower() == ".zip":
        extract_dir = Path(args.out).parent / f"{base.stem}_extracted"
        if not extract_dir.exists():
            print(f"Extracting {base} -> {extract_dir}")
            shutil.unpack_archive(str(base), str(extract_dir))
        base = extract_dir

    root = find_root(base)
    if root is None:
        zips = sorted(base.rglob("*.zip"))
        if zips:
            args.src, args.kaggle_dataset = str(zips[0]), None
            return resolve_source(args)
        raise FileNotFoundError(f"No split.csv or images/ + dataset.xlsx under {base}")
    return root


def read_rows(root: Path, ratios, seed) -> list[dict]:
    csv_path = root / "split.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = [
                {"image_id": r["image_id"], "split": r["split"].strip(), "type": r["type"].strip().lower()}
                for r in csv.DictReader(f)
            ]
        if "normal" not in {r["type"] for r in rows}:
            print("WARNING: no normal image in this dataset; tumor detection cannot be learned")
        return rows
    print(f"No split.csv: stratified split {ratios} with seed {seed}")
    return make_split(root, ratios, seed)


def source_items(root: Path, rows: list[dict]) -> list[tuple[str, Path, Path]]:
    """(split, image path, annotation path) for both supported layouts."""
    split_of = {Path(r["image_id"]).stem: r["split"] for r in rows}
    items = []
    if any((root / s / "images").is_dir() for s in SPLITS):
        dirs = [(s, root / s) for s in SPLITS if (root / s / "images").is_dir()]
    else:
        dirs = [(None, root)]
    for split, folder in dirs:
        for img in sorted((folder / "images").iterdir()):
            s = split or split_of.get(img.stem)
            if s and img.suffix.lower() in IMG_SUFFIXES and img.stem in split_of:
                items.append((s, img, folder / "Annotations" / f"{img.stem}.json"))
    return items


def cache_one(item, out: Path, max_side: int):
    split, img_path, ann_path = item
    out_img = out / split / "images" / f"{img_path.stem}.jpeg"
    out_ann = out / split / "Annotations" / f"{img_path.stem}.json"

    with Image.open(img_path) as im:
        orig_w, orig_h = im.size
        im.draft("RGB", (max_side, max_side))  # fast JPEG downscale
        im = im.convert("L" if im.mode in ("L", "I;16", "I") else "RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        new_w, new_h = im.size
        im.save(out_img, quality=95)

    if ann_path.exists():  # tumor images only
        with open(ann_path, encoding="utf-8") as f:
            ann = json.load(f)
        # Polygons are relative to the size stored in the json (fallback: real size)
        sx = new_w / (ann.get("imageWidth") or orig_w)
        sy = new_h / (ann.get("imageHeight") or orig_h)
        for shape in ann.get("shapes", []):
            shape["points"] = [[x * sx, y * sy] for x, y in shape["points"]]
        ann.update(imageWidth=new_w, imageHeight=new_h, imageData=None, imagePath=out_img.name)
        with open(out_ann, "w", encoding="utf-8") as f:
            json.dump(ann, f)


def main():
    parser = argparse.ArgumentParser(description="Download / split / resize BTXRD")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--kaggle-dataset", help="Kaggle dataset handle, e.g. owner/name")
    source.add_argument("--src", help="dataset folder or .zip")
    parser.add_argument("--out", required=True, help="output folder (use it as --data.path)")
    parser.add_argument("--max-side", type=int, default=1024, help="longest image side after resizing")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.80, 0.15, 0.05))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    out = Path(args.out)
    done = out / f".done_{args.max_side}"
    if done.exists():
        print(f"{out} is already prepared")
    else:
        root = resolve_source(args)
        print(f"Source: {root}")
        rows = read_rows(root, tuple(args.ratios), args.seed)
        items = source_items(root, rows)
        kept = {img.stem for _, img, _ in items}
        rows = [r for r in rows if Path(r["image_id"]).stem in kept]

        for split in SPLITS:
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "Annotations").mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda it: cache_one(it, out, args.max_side), items))

        rows = [{**r, "image_id": f"{Path(r['image_id']).stem}.jpeg"} for r in rows]
        write_csv(rows, out / "split.csv")
        if (root / "dataset.xlsx").exists():
            shutil.copy2(root / "dataset.xlsx", out / "dataset.xlsx")
        done.touch()

    with open(out / "split.csv", newline="", encoding="utf-8") as f:
        print(summary(list(csv.DictReader(f))))


if __name__ == "__main__":
    main()
