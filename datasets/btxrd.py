# ---------------------------------------------------------------
# BTXRD (Bone Tumor X-ray Radiograph Dataset) data module for EoMT.
#
# Bone tumors are cast as an instance segmentation problem with two classes:
#   0 = benign, 1 = malignant
# Every tumor polygon in a LabelMe annotation becomes one instance mask whose
# label is the benign/malignant type of the image. This lets EoMT segment the
# tumor and classify it as benign or malignant with the same query heads.
#
# Expected folder layout (as in the BTXRD release split into train/val/test):
#   <path>/split.csv                     image_id,split,type (benign|malignant)
#   <path>/{train,val,test}/images/*.jpeg
#   <path>/{train,val,test}/Annotations/*.json   (LabelMe format)
# ---------------------------------------------------------------

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import tv_tensors

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms

CLASS_NAMES = ("benign", "malignant")
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Tumor subtypes used as LabelMe labels in BTXRD, mapped to benign/malignant.
# Only used as a fallback when the image type is not found in split.csv.
SUBTYPE_TO_CLASS = {
    "osteochondroma": 0,
    "multiple osteochondromas": 0,
    "simple bone cyst": 0,
    "giant cell tumor": 0,
    "osteofibroma": 0,
    "synovial osteochondroma": 0,
    "other bt": 0,
    "osteosarcoma": 1,
    "other mt": 1,
}

IMG_SUFFIXES = (".jpeg", ".jpg", ".png")


def load_image_types(path: Union[str, Path]) -> dict[str, int]:
    """Read the benign/malignant type of every image from split.csv."""
    csv_path = Path(path, "split.csv")
    types = {}
    if not csv_path.exists():
        return types

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_type = row.get("type", "").strip().lower()
            if image_type in CLASS_TO_ID:
                types[Path(row["image_id"]).stem] = CLASS_TO_ID[image_type]

    return types


def load_tumor_shapes(ann_path: Path) -> tuple[list[tuple[list, str]], int, int]:
    """Return the tumor polygons of a LabelMe file as (points, subtype) pairs.

    Polygons are the segmentation ground truth. Rectangles (bounding boxes) are
    only used for images that have no polygon at all.
    """
    with open(ann_path, encoding="utf-8") as f:
        ann = json.load(f)

    shapes = ann.get("shapes", [])
    polygons = [
        (s["points"], s["label"].strip().lower())
        for s in shapes
        if s.get("shape_type") == "polygon" and len(s["points"]) >= 3
    ]

    if not polygons:
        for s in shapes:
            if s.get("shape_type") == "rectangle" and len(s["points"]) == 2:
                (x0, y0), (x1, y1) = s["points"]
                polygons.append(
                    ([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], s["label"].strip().lower())
                )

    return polygons, ann.get("imageHeight"), ann.get("imageWidth")


class BTXRDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        image_types: dict[str, int],
        transforms: Optional[torch.nn.Module] = None,
        check_empty_targets: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        img_dir = self.root / split / "images"
        ann_dir = self.root / split / "Annotations"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Image folder not found: {img_dir}")

        self.samples: list[tuple[Path, Path, int]] = []
        skipped = Counter()

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_SUFFIXES:
                continue

            ann_path = ann_dir / f"{img_path.stem}.json"
            if not ann_path.exists():
                skipped["no annotation"] += 1
                continue

            polygons, _, _ = load_tumor_shapes(ann_path)
            if check_empty_targets and not polygons:
                skipped["no tumor shape"] += 1
                continue

            cls_id = image_types.get(img_path.stem)
            if cls_id is None and polygons:
                cls_id = SUBTYPE_TO_CLASS.get(polygons[0][1])
            if cls_id is None:
                skipped["unknown type"] += 1
                continue

            self.samples.append((img_path, ann_path, cls_id))

        if skipped:
            logging.info(f"BTXRD/{split}: skipped {dict(skipped)}")

        if not self.samples:
            raise RuntimeError(f"No usable samples found in {self.root / split}")

    @property
    def class_ids(self) -> list[int]:
        return [cls_id for _, _, cls_id in self.samples]

    @staticmethod
    def rasterize(polygons, ann_h, ann_w, height, width) -> list[torch.Tensor]:
        # Annotations may refer to the original resolution; rescale if needed.
        sx = width / ann_w if ann_w else 1.0
        sy = height / ann_h if ann_h else 1.0

        masks = []
        for points, _ in polygons:
            canvas = Image.new("L", (width, height), 0)
            ImageDraw.Draw(canvas).polygon(
                [(x * sx, y * sy) for x, y in points], outline=1, fill=1
            )
            mask = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
            mask = mask.view(height, width).bool()
            if mask.any():
                masks.append(mask)

        return masks

    def __getitem__(self, index: int):
        img_path, ann_path, cls_id = self.samples[index]

        with Image.open(img_path) as pil_img:
            img = tv_tensors.Image(pil_img.convert("RGB"))

        height, width = img.shape[-2:]
        polygons, ann_h, ann_w = load_tumor_shapes(ann_path)
        masks = self.rasterize(polygons, ann_h, ann_w, height, width)

        if masks:
            masks = torch.stack(masks)
        elif self.transforms is not None:
            # Degenerate polygons: training transforms need at least one mask.
            return self[(index + 1) % len(self)]
        else:
            masks = torch.zeros((0, height, width), dtype=torch.bool)

        target = {
            "masks": tv_tensors.Mask(masks),
            "labels": torch.full((len(masks),), cls_id, dtype=torch.long),
            "is_crowd": torch.zeros(len(masks), dtype=torch.bool),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.samples)


class BTXRDInstance(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 8,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 2,
        color_jitter_enabled: bool = True,
        scale_range: tuple[float, float] = (0.5, 1.5),
        check_empty_targets: bool = True,
        balanced_sampling: bool = False,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.balanced_sampling = balanced_sampling
        self.class_names = CLASS_NAMES

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        image_types = load_image_types(self.path)
        dataset_kwargs = {
            "root": self.path,
            "image_types": image_types,
            "check_empty_targets": self.check_empty_targets,
        }

        self.train_dataset = BTXRDDataset(
            split="train", transforms=self.transforms, **dataset_kwargs
        )
        self.val_dataset = BTXRDDataset(split="val", **dataset_kwargs)

        self.test_dataset = None
        if Path(self.path, "test", "images").is_dir():
            self.test_dataset = BTXRDDataset(split="test", **dataset_kwargs)

        return self

    def train_dataloader(self):
        sampler, shuffle = None, True
        if self.balanced_sampling:
            class_ids = self.train_dataset.class_ids
            counts = Counter(class_ids)
            weights = [1.0 / counts[c] for c in class_ids]
            sampler = WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True
            )
            shuffle = False

        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
