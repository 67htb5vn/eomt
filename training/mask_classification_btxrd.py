# ---------------------------------------------------------------
# EoMT for bone tumor segmentation + benign/malignant classification (BTXRD).
#
# Extends MaskClassificationInstance (mask AP per layer) with metrics that
# matter clinically, computed on the final layer:
#   - tumor segmentation: Dice and IoU of the predicted tumor region
#   - image-level classification (malignant = positive): accuracy, balanced
#     accuracy, AUROC, F1, sensitivity, specificity
# It also adds test_step so `python main.py test` works on the test split.
# ---------------------------------------------------------------


from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric.utilities import rank_zero_info
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryRecall,
    BinarySpecificity,
)

from training.mask_classification_instance import MaskClassificationInstance
from training.lightning_module import bold_green, reset


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

        self.cls_metrics = MetricCollection(
            {
                "acc": BinaryAccuracy(),
                "auroc": BinaryAUROC(),
                "f1": BinaryF1Score(),
                "sensitivity": BinaryRecall(),
                "specificity": BinarySpecificity(),
            }
        )
        self.dice_metric = MeanMetric()
        self.iou_metric = MeanMetric()

    @staticmethod
    def predict_btxrd(mask_logits, class_logits, tumor_score_thresh=0.5):
        """Turn the query outputs of one image into BTXRD predictions.

        Args:
            mask_logits: (Q, H, W) mask logits at image resolution.
            class_logits: (Q, num_classes + 1) class logits.

        Returns:
            tumor_mask: (H, W) bool, union of the masks of all confident tumor
                queries (the most confident query is always kept).
            malignant_prob: scalar, P(malignant | tumor) of the top query.
            top_query: index of the most confident tumor query.
        """
        probs = class_logits.float().softmax(dim=-1)
        tumor_scores = probs[:, :-1].sum(dim=-1)
        top_query = tumor_scores.argmax()

        malignant_prob = probs[top_query, 1] / tumor_scores[top_query].clamp_min(1e-6)

        keep = tumor_scores > tumor_score_thresh
        keep[top_query] = True
        tumor_mask = (mask_logits[keep] > 0).any(dim=0)

        return tumor_mask, malignant_prob, top_query

    @torch.compiler.disable
    def update_metrics_btxrd(self, mask_logits, class_logits, targets):
        for j in range(len(mask_logits)):
            tumor_mask, malignant_prob, _ = self.predict_btxrd(
                mask_logits[j], class_logits[j], self.tumor_score_thresh
            )

            gt_mask = targets[j]["masks"].bool().any(dim=0)
            gt_malignant = (targets[j]["labels"] == 1).any().long()

            inter = (tumor_mask & gt_mask).sum().float()
            pred_area = tumor_mask.sum().float()
            gt_area = gt_mask.sum().float()
            union = (tumor_mask | gt_mask).sum().float()

            self.dice_metric.update(2 * inter / (pred_area + gt_area).clamp_min(1))
            self.iou_metric.update(inter / union.clamp_min(1))
            self.cls_metrics.update(malignant_prob[None], gt_malignant[None])

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(transformed_imgs)
        last_layer = len(mask_logits_per_layer) - 1

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, img_sizes
            )

            if i == last_layer:
                self.update_metrics_btxrd(mask_logits, class_logits, targets)

            # Same mask-AP evaluation as MaskClassificationInstance.eval_step
            preds, targets_ = [], []
            for j in range(len(mask_logits)):
                scores = class_logits[j].softmax(dim=-1)[:, :-1]
                labels = (
                    torch.arange(scores.shape[-1], device=self.device)
                    .unsqueeze(0)
                    .repeat(scores.shape[0], 1)
                    .flatten(0, 1)
                )

                topk_scores, topk_indices = scores.flatten(0, 1).topk(
                    min(self.eval_top_k_instances, scores.numel()), sorted=False
                )
                labels = labels[topk_indices]

                topk_indices = topk_indices // scores.shape[-1]
                layer_mask_logits = mask_logits[j][topk_indices]

                masks = layer_mask_logits > 0
                mask_scores = (
                    layer_mask_logits.sigmoid().flatten(1) * masks.flatten(1)
                ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
                scores = topk_scores * mask_scores

                preds.append(dict(masks=masks, labels=labels, scores=scores))
                targets_.append(
                    dict(
                        masks=targets[j]["masks"],
                        labels=targets[j]["labels"],
                        iscrowd=targets[j]["is_crowd"],
                    )
                )

            self.update_metrics_instance(preds, targets_, i)

    def test_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "test")

    def _on_eval_epoch_end_btxrd(self, log_prefix):
        cls_results = self.cls_metrics.compute()
        self.cls_metrics.reset()
        for name, value in cls_results.items():
            self.log(f"metrics/{log_prefix}_cls_{name}", value)

        balanced_acc = (cls_results["sensitivity"] + cls_results["specificity"]) / 2
        self.log(f"metrics/{log_prefix}_cls_balanced_acc", balanced_acc)

        dice = self.dice_metric.compute()
        iou = self.iou_metric.compute()
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.log(f"metrics/{log_prefix}_dice", dice)
        self.log(f"metrics/{log_prefix}_iou", iou)

        # Single number for checkpoint selection: segmentation + classification
        self.log(f"metrics/{log_prefix}_score", (dice + cls_results["auroc"]) / 2)

    def _on_eval_end_btxrd(self, log_prefix):
        if self.trainer.sanity_checking:
            return

        m = self.trainer.callback_metrics
        rank_zero_info(
            f"{bold_green}Dice: {m[f'metrics/{log_prefix}_dice'] * 100:.1f} | "
            f"IoU: {m[f'metrics/{log_prefix}_iou'] * 100:.1f} | "
            f"Acc: {m[f'metrics/{log_prefix}_cls_acc'] * 100:.1f} | "
            f"Bal. Acc: {m[f'metrics/{log_prefix}_cls_balanced_acc'] * 100:.1f} | "
            f"AUROC: {m[f'metrics/{log_prefix}_cls_auroc'] * 100:.1f} | "
            f"Sens: {m[f'metrics/{log_prefix}_cls_sensitivity'] * 100:.1f} | "
            f"Spec: {m[f'metrics/{log_prefix}_cls_specificity'] * 100:.1f}{reset}"
        )

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_instance("val")
        self._on_eval_epoch_end_btxrd("val")

    def on_validation_end(self):
        self._on_eval_end_instance("val")
        self._on_eval_end_btxrd("val")

    def on_test_epoch_end(self):
        self._on_eval_epoch_end_instance("test")
        self._on_eval_epoch_end_btxrd("test")

    def on_test_end(self):
        self._on_eval_end_instance("test")
        self._on_eval_end_btxrd("test")
