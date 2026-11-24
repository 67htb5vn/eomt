# ---------------------------------------------------------------
# Checkpoint selection and early stopping that start after mask annealing.
#
# During mask annealing, validation still uses part of the masked attention
# that inference turns off (masked_attn_enabled: false), so a validation score
# from that phase does not describe the model as it is used at test time.
# These two callbacks are Lightning's ModelCheckpoint / EarlyStopping, except
# that they ignore the validation results of epochs before `start_epoch`
# (0-based; the first epoch that ends with masked attention fully off):
#   - the best checkpoint is chosen among epochs >= start_epoch only
#     (last.ckpt is still saved every epoch, for resuming);
#   - patience is counted from start_epoch on, so training cannot stop
#     before start_epoch + patience.
# ---------------------------------------------------------------


from typing import Optional

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


class ModelCheckpointFrom(ModelCheckpoint):
    def __init__(
        self,
        start_epoch: int = 0,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        mode: str = "min",
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        auto_insert_metric_name: bool = True,
        verbose: bool = False,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_last=save_last,
            save_top_k=save_top_k,
            auto_insert_metric_name=auto_insert_metric_name,
            verbose=verbose,
        )
        self.start_epoch = start_epoch

    def _save_topk_checkpoint(self, trainer, monitor_candidates):
        if trainer.current_epoch < self.start_epoch:
            return
        super()._save_topk_checkpoint(trainer, monitor_candidates)


class EarlyStoppingFrom(EarlyStopping):
    def __init__(
        self,
        monitor: str,
        start_epoch: int = 0,
        patience: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__(
            monitor=monitor,
            patience=patience,
            mode=mode,
            min_delta=min_delta,
            verbose=verbose,
        )
        self.start_epoch = start_epoch

    def _run_early_stopping_check(self, trainer):
        if trainer.current_epoch < self.start_epoch:
            return
        super()._run_early_stopping_check(trainer)
