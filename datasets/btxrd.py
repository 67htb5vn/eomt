# ---------------------------------------------------------------
# BTXRD (Bone Tumor X-ray Radiograph Dataset) data module for EoMT.
#
# Tumor detection, segmentation and benign/malignant classification are cast
# as one instance segmentation problem with two mask classes:
#   0 = benign, 1 = malignant   (+ the implicit "no object" class of EoMT)
#
# - Tumor image: every tumor polygon in its LabelMe file becomes one instance
#   mask, labelled with the benign/malignant type of the image.
# - Normal image: no annotation, so the target has zero masks. EoMT then learns
#   to send all of its queries to "no object" for this image.
# At inference, an image contains a tumor if at least one query is confident
# enough; that query's masks give the tumor region and its class gives the
# benign/malignant label. Detection and segmentation happen in a single pass.
#
# Expected folder layout (see tools/split_btxrd.py):
#   <path>/split.csv                     image_id,split,type (normal|benign|malignant)
#   <path>/{train,val,test}/images/*.jpeg
#   <path>/{train,val,test}/Annotations/*.json   (LabelMe, tumor images only)
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
from torchvision.tv_tensors import wrap

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms

# Mask classes predicted by EoMT (the "no object" class is added by the model).
CLASS_NAMES = ("benign", "malignant")

# Image-level labels: what the whole pipeline should output for one radiograph.
IMAGE_CLASSES = ("normal", "benign", "malignant")
IMAGE_CLASS_TO_ID = {name: i for i, name in enumerate(IMAGE_CLASSES)}
NORMAL = IMAGE_CLASS_TO_ID["normal"]

# Tumor subtypes used as LabelMe labels in BTXRD, mapped to image-level labels.
# Only used as a fallback when the image type is not found in split.csv.
SUBTYPE_TO_IMAGE_CLASS = {
    "osteochondroma": 1,
    "multiple osteochondromas": 1,
    "simple bone cyst": 1,
    "giant cell tumor": 1,
    "osteofibroma": 1,
    "synovial osteochondroma": 1,
    "other bt": 1,
    "osteosarcoma": 2,
    "other mt": 2,
}

IMG_SUFFIXES = (".jpeg", ".jpg", ".png")


def mask_class(image_class: int) -> int:
    """Image-level label (1 benign, 2 malignant) -> EoMT mask class (0, 1)."""
    return image_class - 1


def load_image_types(path: Union[str, Path]) -> dict[str, int]:
    """Read the image-level type (normal/benign/malignant) of every image from split.csv."""
    csv_path = Path(path, "split.csv")
    types = {}
    if not csv_path.exists():
        return types

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_type = row.get("type", "").strip().lower()
            if image_type in IMAGE_CLASS_TO_ID:
                types[Path(row["image_id"]).stem] = IMAGE_CLASS_TO_ID[image_type]

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


class BTXRDTransforms(Transforms):
    """Project training transforms, extended to images without any mask.

    The base class needs at least one mask (it retries the random crop until a
    mask survives, and torchvision cannot resize/pad a zero-sized mask tensor).
    For a normal image we run the same geometric transforms with a full-image
    dummy mask, then drop it, so the image is augmented in exactly the same way.
    """

    def forward(self, img, target):
        if len(target["masks"]) > 0:
            return super().forward(img, target)

        height, width = img.shape[-2:]
        dummy = {
            "masks": tv_tensors.Mask(torch.ones((1, height, width), dtype=torch.bool)),
            "labels": torch.zeros(1, dtype=torch.long),
            "is_crowd": torch.zeros(1, dtype=torch.bool),
        }
        img, dummy = super().forward(img, dummy)

        return img, {k: wrap(v[:0], like=v) for k, v in dummy.items()}


class BTXRDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        image_types: dict[str, int],
        transforms: Optional[torch.nn.Module] = None,
        include_normal: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        img_dir = self.root / split / "images"
        ann_dir = self.root / split / "Annotations"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Image folder not found: {img_dir}")

        # (image path, annotation path or None, image-level label)
        self.samples: list[tuple[Path, Optional[Path], int]] = []
        skipped = Counter()

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_SUFFIXES:
                continue

            ann_path = ann_dir / f"{img_path.stem}.json"
            polygons = load_tumor_shapes(ann_path)[0] if ann_path.exists() else []

            image_class = image_types.get(img_path.stem)
            if image_class is None and polygons:
                image_class = SUBTYPE_TO_IMAGE_CLASS.get(polygons[0][1])
            if image_class is None:
                skipped["unknown type"] += 1
                continue

            if image_class == NORMAL:
                if include_normal:
                    self.samples.append((img_path, None, NORMAL))
                else:
                    skipped["normal (excluded)"] += 1
                continue

            if not polygons:
                skipped["tumor without polygon"] += 1
                continue

            self.samples.append((img_path, ann_path, image_class))

        if skipped:
            logging.info(f"BTXRD/{split}: skipped {dict(skipped)}")

        if not self.samples:
            raise RuntimeError(f"No usable samples found in {self.root / split}")

    @property
    def class_ids(self) -> list[int]:
        """Image-level label of every sample (0 normal, 1 benign, 2 malignant)."""
        return [image_class for _, _, image_class in self.samples]

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
        img_path, ann_path, image_class = self.samples[index]

        with Image.open(img_path) as pil_img:
            img = tv_tensors.Image(pil_img.convert("RGB"))

        height, width = img.shape[-2:]
        masks = []
        if ann_path is not None:
            polygons, ann_h, ann_w = load_tumor_shapes(ann_path)
            masks = self.rasterize(polygons, ann_h, ann_w, height, width)
            if not masks and self.transforms is not None:
                # Tumor image whose polygons are degenerate: do not train on it
                # as if it were normal.
                return self[(index + 1) % len(self)]

        if masks:
            masks = torch.stack(masks)
        else:
            masks = torch.zeros((0, height, width), dtype=torch.bool)

        labels = (
            torch.full((len(masks),), mask_class(image_class), dtype=torch.long)
            if image_class != NORMAL
            else torch.zeros(0, dtype=torch.long)
        )
        target = {
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "is_crowd": torch.zeros(len(masks), dtype=torch.bool),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # Image-level ground truth for the detection / classification metrics.
        # Added after the transforms, which filter every per-mask entry.
        target["image_label"] = torch.tensor(image_class, dtype=torch.long)

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
        check_empty_targets: bool = False,
        balanced_sampling: bool = False,
    ) -> None:
        """
        Args:
            check_empty_targets: True drops the normal images (no mask), which
                turns the task back into segmentation + classification of
                tumor images only. Keep False for detection + segmentation.
            balanced_sampling: sample normal / benign / malignant images with
                equal probability during training.
        """
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
        self.image_class_names = IMAGE_CLASSES

        self.transforms = BTXRDTransforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        image_types = load_image_types(self.path)
        dataset_kwargs = {
            "root": self.path,
            "image_types": image_types,
            "include_normal": not self.check_empty_targets,
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
