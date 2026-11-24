# ---------------------------------------------------------------
# EoMT for bone tumor detection + segmentation + benign/malignant
# classification on BTXRD, in a single forward pass.
#
# Model, loss, optimizer, schedule and mask-AP evaluation are the project's own
# (MaskClassificationInstance). Normal radiographs are targets with zero masks:
# the project's loss then trains all queries towards "no object".
#
# This subclass only adds the BTXRD read-out of the final layer and its metrics.
# For one radiograph:
#   tumor score  = max over queries of P(benign) + P(malignant)
#   has tumor    = tumor score > tumor_score_thresh
#   tumor region = union of the masks of all queries above the threshold
#                  (empty when no tumor is detected)
#   P(malignant) = P(malignant) / P(tumor) of the most confident query
# Metrics:
#   - detection (tumor = positive, all images): acc, balanced acc, AUROC, F1,
#     sensitivity, specificity
#   - segmentation: Dice / IoU on tumor images, Dice over all images
#     (an empty prediction on a normal image counts as Dice 1)
#   - benign/malignant classification (malignant = positive, tumor images)
#   - 3-class image accuracy (normal / benign / malignant)
# It also adds test_step so `python main.py test` works on the test split.
#
# val_mask_ap=False skips the project's mask AP during validation only (it is
# not part of metrics/val_score, which selects the checkpoint, and encoding
# 100 full-resolution masks x 4 layers per image on the CPU dominates the
# validation time). The forward pass and all BTXRD metrics are unchanged, and
# `main.py test` still computes mask AP.
# ---------------------------------------------------------------


from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric.utilities import rank_zero_info
from torchmetrics import MeanMetric, MetricCollection, SumMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryRecall,
    BinarySpecificity,
    MulticlassAccuracy,
)

from training.mask_classification_instance import MaskClassificationInstance
from training.lightning_module import bold_green, reset


def binary_metrics() -> MetricCollection:
    return MetricCollection(
        {
            "acc": BinaryAccuracy(),
            "auroc": BinaryAUROC(),
            "f1": BinaryF1Score(),
            "sensitivity": BinaryRecall(),
            "specificity": BinarySpecificity(),
        }
    )


class MaskClassificationBTXRD(MaskClassificationInstance):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        eval_top_k_instances: int = 100,
        tumor_score_thresh: float = 0.5,
        malignant_thresh: float = 0.5,
        val_mask_ap: bool = True,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            no_object_coefficient=no_object_coefficient,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            mask_thresh=mask_thresh,
            overlap_thresh=overlap_thresh,
            eval_top_k_instances=eval_top_k_instances,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        if num_classes != 2:
            raise ValueError("BTXRD expects num_classes=2 (benign, malignant)")

        self.tumor_score_thresh = tumor_score_thresh
        self.malignant_thresh = malignant_thresh
        self.val_mask_ap = val_mask_ap

        self.det_metrics = binary_metrics()  # tumor vs normal, all images
        self.cls_metrics = binary_metrics()  # malignant vs benign, tumor images
        self.img_metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=3, average="micro"),
                "balanced_acc": MulticlassAccuracy(num_classes=3, average="macro"),
            }
        )
        self.dice_metric = MeanMetric()  # tumor images
        self.iou_metric = MeanMetric()  # tumor images
        self.dice_all_metric = MeanMetric()  # all images
        self.num_tumor_images = SumMetric()

    def on_fit_start(self):
        # Annealing steps left empty in the config: derive them from the real
        # number of training steps (which depends on dataset size, batch size,
        # number of GPUs and epochs), with the proportions of the project's
        # COCO/ADE20K configs: one start per masked block spread over
        # [1/6, 4/6] of training, each lasting 1/6 of training.
        if self.attn_mask_annealing_enabled and not self.attn_mask_annealing_start_steps:
            total_steps = self.trainer.estimated_stepping_batches
            num_blocks = self.network.num_blocks
            unit = total_steps / 6
            starts = [
                round(unit + k * 3 * unit / max(1, num_blocks - 1))
                for k in range(num_blocks)
            ]
            self.attn_mask_annealing_start_steps = starts
            self.attn_mask_annealing_end_steps = [round(s + unit) for s in starts]
            rank_zero_info(
                f"Mask annealing over {total_steps} steps: starts {starts}, "
                f"ends {self.attn_mask_annealing_end_steps}"
            )

    def forward(self, imgs):
        mask_logits_per_layer, class_logits_per_layer = super().forward(imgs)
        if not self.training:
            # Final-layer outputs, reused by eval_step after the project's evaluation
            self._eval_outputs = (mask_logits_per_layer[-1], class_logits_per_layer[-1])
        return mask_logits_per_layer, class_logits_per_layer

    @staticmethod
    def predict_btxrd(
        mask_logits, class_logits, tumor_score_thresh=0.5, malignant_thresh=0.5
    ):
        """Turn the query outputs of one image into BTXRD predictions.

        Args:
            mask_logits: (Q, H, W) mask logits at image resolution.
            class_logits: (Q, num_classes + 1) class logits.

        Returns:
            tumor_mask: (H, W) bool, union of the masks of the queries whose
                tumor score exceeds the threshold; all False if none does.
            tumor_prob: scalar, tumor score of the most confident query
                (image-level detection score).
            malignant_prob: scalar, P(malignant | tumor) of that query.
            image_pred: 0 normal, 1 benign, 2 malignant.
        """
        probs = class_logits.float().softmax(dim=-1)
        tumor_scores = probs[:, :-1].sum(dim=-1)
        top_query = tumor_scores.argmax()
        tumor_prob = tumor_scores[top_query]

        malignant_prob = probs[top_query, 1] / tumor_prob.clamp_min(1e-6)

        keep = tumor_scores > tumor_score_thresh
        tumor_mask = (mask_logits[keep] > 0).any(dim=0)

        image_pred = torch.where(
            tumor_prob > tumor_score_thresh,
            1 + (malignant_prob >= malignant_thresh).long(),
            torch.zeros_like(top_query),
        )

        return tumor_mask, tumor_prob, malignant_prob, image_pred

    @torch.compiler.disable
    def update_metrics_btxrd(self, mask_logits, class_logits, targets):
        for j in range(len(mask_logits)):
            tumor_mask, tumor_prob, malignant_prob, image_pred = self.predict_btxrd(
                mask_logits[j],
                class_logits[j],
                self.tumor_score_thresh,
                self.malignant_thresh,
            )

            image_label = targets[j]["image_label"]
            has_tumor = image_label > 0
            gt_mask = targets[j]["masks"].bool().any(dim=0)

            self.det_metrics.update(tumor_prob[None], has_tumor.long()[None])
            self.img_metrics.update(image_pred[None], image_label[None])

            inter = (tumor_mask & gt_mask).sum().float()
            area_sum = (tumor_mask.sum() + gt_mask.sum()).float()
            union = (tumor_mask | gt_mask).sum().float()
            dice = 2 * inter / area_sum if area_sum > 0 else torch.ones_like(inter)
            self.dice_all_metric.update(dice)

            if has_tumor and gt_mask.any():
                self.dice_metric.update(dice)
                self.iou_metric.update(inter / union.clamp_min(1))
                self.cls_metrics.update(
                    malignant_prob[None], (image_label == 2).long()[None]
                )
                self.num_tumor_images.update(1.0)

    def _skip_mask_ap(self, log_prefix):
        return log_prefix == "val" and not self.val_mask_ap

    def eval_step(self, batch, batch_idx=None, log_prefix=None):
        imgs, targets = batch
        if self._skip_mask_ap(log_prefix):
            # Same forward pass as the project's eval_step, without mask AP
            self(self.resize_and_pad_imgs_instance_panoptic(imgs))
        else:
            # Project's evaluation, unchanged: forward pass + mask AP for every layer
            super().eval_step(batch, batch_idx, log_prefix)

        # BTXRD read-out of the same forward pass (final layer)
        mask_logits, class_logits = self._eval_outputs
        mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
        mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
            mask_logits, [img.shape[-2:] for img in imgs]
        )
        self.update_metrics_btxrd(mask_logits, class_logits, targets)

    def test_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "test")

    def _log_binary(self, metrics, name, log_prefix, valid=True):
        results = metrics.compute() if valid else {}
        metrics.reset()
        nan = torch.tensor(float("nan"), device=self.device)
        for key in ("acc", "auroc", "f1", "sensitivity", "specificity"):
            self.log(f"metrics/{log_prefix}_{name}_{key}", results.get(key, nan))
        balanced = (
            (results["sensitivity"] + results["specificity"]) / 2 if valid else nan
        )
        self.log(f"metrics/{log_prefix}_{name}_balanced_acc", balanced)
        return results.get("auroc", nan)

    def _on_eval_epoch_end_btxrd(self, log_prefix):
        # SumMetric.compute is synced across ranks, so every rank takes the
        # same branch below (metric computes are collective under DDP).
        has_tumor_images = bool(self.num_tumor_images.compute() > 0)
        self.num_tumor_images.reset()

        det_auroc = self._log_binary(self.det_metrics, "det", log_prefix)
        cls_auroc = self._log_binary(
            self.cls_metrics, "cls", log_prefix, valid=has_tumor_images
        )

        img_results = self.img_metrics.compute()
        self.img_metrics.reset()
        for key, value in img_results.items():
            self.log(f"metrics/{log_prefix}_img3_{key}", value)

        dice_all = self.dice_all_metric.compute()
        self.dice_all_metric.reset()
        self.log(f"metrics/{log_prefix}_dice_all", dice_all)

        if has_tumor_images:
            dice = self.dice_metric.compute()
            iou = self.iou_metric.compute()
        else:
            dice = iou = torch.tensor(float("nan"), device=self.device)
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.log(f"metrics/{log_prefix}_dice", dice)
        self.log(f"metrics/{log_prefix}_iou", iou)

        # Single number for checkpoint selection: detection + segmentation +
        # benign/malignant classification
        self.log(f"metrics/{log_prefix}_score", (det_auroc + dice + cls_auroc) / 3)

    def _on_eval_end_btxrd(self, log_prefix):
        if self.trainer.sanity_checking:
            return

        m = self.trainer.callback_metrics
        p = f"metrics/{log_prefix}"
        rank_zero_info(
            f"{bold_green}Detection  | AUROC: {m[f'{p}_det_auroc'] * 100:.1f} | "
            f"Sens: {m[f'{p}_det_sensitivity'] * 100:.1f} | "
            f"Spec: {m[f'{p}_det_specificity'] * 100:.1f} | "
            f"Bal. Acc: {m[f'{p}_det_balanced_acc'] * 100:.1f}\n"
            f"Segmentation | Dice: {m[f'{p}_dice'] * 100:.1f} | "
            f"IoU: {m[f'{p}_iou'] * 100:.1f} | "
            f"Dice (all images): {m[f'{p}_dice_all'] * 100:.1f}\n"
            f"Benign/Malignant | AUROC: {m[f'{p}_cls_auroc'] * 100:.1f} | "
            f"Sens: {m[f'{p}_cls_sensitivity'] * 100:.1f} | "
            f"Spec: {m[f'{p}_cls_specificity'] * 100:.1f} | "
            f"Bal. Acc: {m[f'{p}_cls_balanced_acc'] * 100:.1f}\n"
            f"3-class (normal/benign/malignant) | Acc: {m[f'{p}_img3_acc'] * 100:.1f} | "
            f"Bal. Acc: {m[f'{p}_img3_balanced_acc'] * 100:.1f}{reset}"
        )

    def on_validation_epoch_end(self):
        if not self._skip_mask_ap("val"):
            self._on_eval_epoch_end_instance("val")
        self._on_eval_epoch_end_btxrd("val")

    def on_validation_end(self):
        if not self._skip_mask_ap("val"):
            self._on_eval_end_instance("val")
        self._on_eval_end_btxrd("val")

    def on_test_epoch_end(self):
        self._on_eval_epoch_end_instance("test")
        self._on_eval_epoch_end_btxrd("test")

    def on_test_end(self):
        self._on_eval_end_instance("test")
        self._on_eval_end_btxrd("test")
