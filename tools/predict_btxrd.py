# ---------------------------------------------------------------
# Inference and per-image evaluation of a trained BTXRD EoMT model.
#
# One forward pass per radiograph gives: tumor or not (detection), the tumor
# region (segmentation) and benign / malignant (classification), using
# MaskClassificationBTXRD.predict_btxrd, the same read-out as the training metrics.
# The model is built from the config as in inference.ipynb.
#
# Modes:
#   --data-path DIR --split test   a prepared dataset split: predictions + metrics
#                                  (reports, confusion matrix) + figures
#   --images A.jpeg B.png DIR/     new radiographs: predictions + figures
#
# Outputs in --out: predictions.csv, metrics.json + report.txt (dataset mode),
# confusion_matrix.png, roc.png and vis/*.png (image | ground truth | prediction).
#
# Usage:
#   python tools/predict_btxrd.py -c configs/dinov2/btxrd/instance/eomt_base_640.yaml \
#     --weights best.bin --data-path /content/btxrd_cache --split test --out results/test
#   python tools/predict_btxrd.py -c configs/dinov2/btxrd/instance/eomt_base_640.yaml \
#     --weights best.bin --images xray.jpeg --out results/new
# ---------------------------------------------------------------

import argparse
import importlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.btxrd import IMAGE_CLASSES, IMG_SUFFIXES, BTXRDDataset, load_image_types  # noqa: E402

warnings.filterwarnings("ignore", message=r".*is an instance of `nn\.Module` and is already saved.*")

COLORS = {1: (0.1, 0.8, 0.2), 2: (0.9, 0.1, 0.1)}  # benign green, malignant red


def merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cls(path: str):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def build_model(config: dict, weights: str, device):
    """As in inference.ipynb: encoder -> network (masked attention off) -> Lightning module."""
    data_args = config["data"]["init_args"]
    img_size = tuple(data_args["img_size"])
    num_classes = data_args.get("num_classes", 2)

    net_cfg = config["model"]["init_args"]["network"]
    enc_cfg = net_cfg["init_args"]["encoder"]
    encoder = load_cls(enc_cfg["class_path"])(
        img_size=img_size, ckpt_path=weights, **enc_cfg.get("init_args", {})
    )
    net_kwargs = {k: v for k, v in net_cfg["init_args"].items() if k not in ("encoder", "masked_attn_enabled")}
    network = load_cls(net_cfg["class_path"])(
        encoder=encoder, num_classes=num_classes, masked_attn_enabled=False, **net_kwargs
    )
    model_kwargs = {
        k: v for k, v in config["model"]["init_args"].items()
        if k not in ("network", "ckpt_path", "load_ckpt_class_head", "delta_weights")
    }
    model = load_cls(config["model"]["class_path"])(
        network=network, img_size=img_size, num_classes=num_classes,
        ckpt_path=weights, load_ckpt_class_head=True, **model_kwargs,
    )
    return model.eval().to(device)


@torch.no_grad()
def predict(model, img: torch.Tensor, device):
    """img: uint8 (3, H, W) -> tumor mask (H, W), P(tumor), P(malignant), class 0/1/2."""
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        x = model.resize_and_pad_imgs_instance_panoptic([img.to(device)])
        mask_logits_per_layer, class_logits_per_layer = model(x)
        mask_logits = F.interpolate(mask_logits_per_layer[-1], model.img_size, mode="bilinear")
        mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(mask_logits, [img.shape[-2:]])[0]
        mask, p_tumor, p_mal, pred = model.predict_btxrd(
            mask_logits.float(), class_logits_per_layer[-1][0], model.tumor_score_thresh, model.malignant_thresh
        )
    return mask.cpu(), float(p_tumor), float(p_mal), int(pred)


def load_image(path: Path, max_side: int) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        return torch.from_numpy(np.array(im)).permute(2, 0, 1).contiguous()


def save_figure(path: Path, img, gt_mask, gt_label, pred_mask, row):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = img.permute(1, 2, 0).numpy()
    panels = [("Image", None, None)]
    if gt_label is not None:
        panels.append((f"Ground truth: {IMAGE_CLASSES[gt_label]}", gt_mask, gt_label))
    title = f"Prediction: {IMAGE_CLASSES[row['pred']]}  P(tumor)={row['p_tumor']:.2f}"
    if row["pred"] > 0:
        title += f"  P(malignant)={row['p_malignant']:.2f}"
    if row.get("dice") is not None and not np.isnan(row["dice"]):
        title += f"  Dice={row['dice']:.2f}"
    panels.append((title, pred_mask, row["pred"]))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.5))
    for ax, (t, mask, cls) in zip(axes, panels):
        ax.imshow(base)
        if mask is not None and cls and mask.any():
            rgba = np.zeros((*mask.shape, 4))
            rgba[mask.numpy()] = (*COLORS[cls], 0.45)
            ax.imshow(rgba)
        if gt_mask is not None and gt_mask.any():
            ax.contour(gt_mask.numpy(), levels=[0.5], colors="yellow", linewidths=1)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{row['image']}  (green = benign, red = malignant, yellow = ground truth)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def evaluate(df, out: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        RocCurveDisplay,
        balanced_accuracy_score,
        classification_report,
        roc_auc_score,
    )

    tumor, normal = df[df.label > 0], df[df.label == 0]
    is_mal = (tumor.label == 2).astype(int)
    metrics = {
        "n_images": len(df), "n_normal": len(normal), "n_tumor": len(tumor),
        "det_sensitivity": float((tumor.pred > 0).mean()) if len(tumor) else None,
        "det_specificity": float((normal.pred == 0).mean()) if len(normal) else None,
        "det_auroc": float(roc_auc_score(df.label > 0, df.p_tumor)) if len(tumor) and len(normal) else None,
        "dice": float(tumor.dice.mean()) if len(tumor) else None,
        "iou": float(tumor.iou.mean()) if len(tumor) else None,
        "cls_auroc": float(roc_auc_score(is_mal, tumor.p_malignant)) if is_mal.nunique() == 2 else None,
        "cls_balanced_acc": float(balanced_accuracy_score(is_mal, tumor.p_malignant >= 0.5)) if len(tumor) else None,
        "img3_acc": float((df.label == df.pred).mean()),
        "img3_balanced_acc": float(balanced_accuracy_score(df.label, df.pred)),
    }

    report = [
        "=== Tumor detection (all images, tumor = positive) ===",
        classification_report(df.label > 0, df.pred > 0, labels=[False, True],
                              target_names=["normal", "tumor"], digits=4, zero_division=0),
        "=== Benign / malignant (tumor images, malignant = positive) ===",
        classification_report(is_mal, (tumor.p_malignant >= 0.5).astype(int), labels=[0, 1],
                              target_names=["benign", "malignant"], digits=4, zero_division=0),
        "=== Final output, 3 classes ===",
        classification_report(df.label, df.pred, labels=[0, 1, 2],
                              target_names=list(IMAGE_CLASSES), digits=4, zero_division=0),
        "=== Summary ===",
        json.dumps(metrics, indent=2),
    ]
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\n".join(report))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(df.label, df.pred, labels=[0, 1, 2],
                                            display_labels=IMAGE_CLASSES, cmap="Blues", ax=ax)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=100)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    if metrics["det_auroc"] is not None:
        RocCurveDisplay.from_predictions(df.label > 0, df.p_tumor, name="EoMT", ax=axes[0])
    axes[0].set_title("ROC tumor detection")
    if metrics["cls_auroc"] is not None:
        RocCurveDisplay.from_predictions(is_mal, tumor.p_malignant, name="EoMT", ax=axes[1])
    axes[1].set_title("ROC benign / malignant")
    for ax in axes:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=100)
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="BTXRD EoMT inference / evaluation")
    parser.add_argument("-c", "--config", action="append", required=True,
                        help="training config(s); later ones override earlier ones")
    parser.add_argument("--weights", required=True, help=".bin state dict (tools/export_weights.py)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-path", help="prepared dataset folder (tools/prepare_btxrd.py)")
    source.add_argument("--images", nargs="+", help="image files or folders")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-vis", type=int, default=6,
                        help="figures to save: random images + this many of each error type")
    parser.add_argument("--max-side", type=int, default=1024, help="resize new images like the prepared data")
    parser.add_argument("--tumor-thresh", type=float, help="override tumor_score_thresh")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = {}
    for path in args.config:
        with open(path) as f:
            config = merge(config, yaml.safe_load(f))
    device = torch.device(args.device)
    model = build_model(config, args.weights, device)
    if args.tumor_thresh is not None:
        model.tumor_score_thresh = args.tumor_thresh

    out = Path(args.out)
    (out / "vis").mkdir(parents=True, exist_ok=True)

    import pandas as pd

    rows, cache = [], {}
    if args.data_path:
        ds = BTXRDDataset(args.data_path, args.split, load_image_types(args.data_path))
        for idx in range(len(ds)):
            img, tgt = ds[idx]
            mask, p_tumor, p_mal, pred = predict(model, img, device)
            gt = tgt["masks"].bool().any(0)
            inter = (mask & gt).sum().item()
            has_gt = bool(gt.any())
            rows.append(dict(
                image=ds.samples[idx][0].stem, label=int(tgt["image_label"]), pred=pred,
                p_tumor=p_tumor, p_malignant=p_mal,
                dice=2 * inter / max(1, mask.sum().item() + gt.sum().item()) if has_gt else np.nan,
                iou=inter / max(1, (mask | gt).sum().item()) if has_gt else np.nan,
            ))
            cache[idx] = mask
            if (idx + 1) % 50 == 0:
                print(f"{idx + 1}/{len(ds)}")
    else:
        paths = []
        for p in map(Path, args.images):
            paths += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_SUFFIXES) if p.is_dir() else [p]
        for idx, p in enumerate(paths):
            img = load_image(p, args.max_side)
            mask, p_tumor, p_mal, pred = predict(model, img, device)
            row = dict(image=p.stem, pred=pred, p_tumor=p_tumor, p_malignant=p_mal)
            rows.append(row)
            save_figure(out / "vis" / f"{p.stem}.png", img, None, None, mask, row)

    df = pd.DataFrame(rows)
    df.insert(2, "pred_name", df.pred.map(dict(enumerate(IMAGE_CLASSES))))
    if "label" in df:
        df.insert(2, "label_name", df.label.map(dict(enumerate(IMAGE_CLASSES))))
    df.to_csv(out / "predictions.csv", index=False)
    print(f"Predictions: {out / 'predictions.csv'}")

    if args.data_path:
        evaluate(df, out)
        k = args.num_vis
        rng = np.random.default_rng(0)
        groups = {
            "random": list(rng.choice(len(df), size=min(k, len(df)), replace=False)),
            "missed": df.index[(df.label > 0) & (df.pred == 0)].tolist()[:k],
            "false_alarm": df.index[(df.label == 0) & (df.pred > 0)].tolist()[:k],
            "swapped": df.index[(df.label > 0) & (df.pred > 0) & (df.label != df.pred)].tolist()[:k],
            "low_dice": df[df.label > 0].nsmallest(k, "dice").index.tolist(),
        }
        for group, indices in groups.items():
            for idx in indices:
                img, tgt = ds[idx]
                save_figure(out / "vis" / f"{group}_{df.image[idx]}.png", img,
                            tgt["masks"].bool().any(0), int(tgt["image_label"]), cache[idx], df.iloc[idx])
    print(f"Figures: {out / 'vis'}")


if __name__ == "__main__":
    main()
