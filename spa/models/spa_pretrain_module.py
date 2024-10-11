from __future__ import annotations

from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule

from spa import utils as U
from spa.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SPAPretrainModule(LightningModule):
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        train_metrics,
        val_metrics,
        best_val_metrics,
        compile: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["model", "train_metrics", "val_metrics", "best_val_metrics"],
        )

        self.model = model

        # metric objects for calculating and averaging accuracy across batches
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

        # for tracking best so far validation metrics
        self.best_val_metrics = best_val_metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.best_val_metrics.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(batch)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.model.train()

    def training_step(
        self,
        batch,
        batch_idx,
    ) -> torch.Tensor:
        if not isinstance(batch, list):
            batch = [batch]

        count = 0
        dst_batch = None
        dataloader_idx = 0
        for i, x in enumerate(batch):
            if x is not None:
                count += 1
                dst_batch = x
                dataloader_idx = i
        batch = dst_batch
        assert count == 1, "Only one dataset is allowed for each iteration"

        loss_dict = self.model_step(batch)

        # update and log metrics
        self.train_metrics(loss_dict)
        batch_size = len(batch["img"])
        self.log_dict(
            self.train_metrics.metrics_dict(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # log global_step
        self.log(
            "global_step",
            self.global_step,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=False,
        )

        # return loss or backpropagation will fail
        return loss_dict["loss"]

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.model.eval()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss_dict = self.model_step(batch)

        # update and log metrics
        self.val_metrics(loss_dict)
        batch_size = len(batch["img"])
        self.log_dict(
            self.val_metrics.metrics_dict(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # check input
        def project(xyz, K, R, T):
            xyz = np.dot(xyz, R.T) + T.T
            xyz = np.dot(xyz, K.T)
            xy = xyz[:, :2] / xyz[:, 2:]
            return xy

        # visualize
        gt_img = loss_dict["gt_img"]
        pred_img = loss_dict["pred_img"]
        gt_depth = loss_dict["gt_depth"]
        pred_depth = loss_dict["pred_depth"]
        if "pred_normal" not in loss_dict:
            pred_normal = np.zeros_like(gt_img)
        else:
            pred_normal = loss_dict["pred_normal"]

        img_tfboard = np.concatenate(
            [gt_img, pred_img, pred_normal], axis=1
        )  # (H, W, 3)
        img_tfboard /= 255.0

        self.logger.experiment.add_images(
            f"val/img_tfboard_{batch_idx}",
            img_tfboard,
            global_step=self.current_epoch,
            dataformats="HWC",
        )

        def min_max_normalize(data):
            return (
                (data - data[data > 0].min())
                / (data[data > 0].max() - data[data > 0].min())
            ).clip(0.0, 1.0)

        depth_tfboard = np.concatenate([gt_depth, pred_depth], axis=1)
        depth_tfboard = cm.bwr(min_max_normalize(depth_tfboard / 2.0).clip(0.0, 1.0))
        # depth_tfboard = depth_tfboard.transpose(2, 0, 1)
        self.logger.experiment.add_images(
            f"val/depth_tfboard_{batch_idx}",
            depth_tfboard,
            global_step=self.current_epoch,
            dataformats="HWC",
        )

        if "similarity" in loss_dict:
            similarity = loss_dict["similarity"]
            self.logger.experiment.add_images(
                f"val/semantic_similarity_map_{batch_idx}",
                similarity,
                global_step=self.current_epoch,
                dataformats="HW",
            )
        if "semantic_gt_pca" in loss_dict:
            semantic_gt_pca = loss_dict["semantic_gt_pca"]
            semantic_pred_pca = loss_dict["semantic_pred_pca"]

            semantic_pca = np.concatenate(
                [
                    semantic_gt_pca,
                    semantic_pred_pca,
                ],
                axis=1,
            )
            self.logger.experiment.add_images(
                f"val/semantic_pca_{batch_idx}",
                semantic_pca,
                global_step=self.current_epoch,
                dataformats="HWC",
            )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        metrics = self.val_metrics.compute()  # get current val metrics
        self.best_val_metrics(metrics)  # update best so far val metrics
        # log `best_val_metrics` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log_dict(self.best_val_metrics.compute(), sync_dist=True, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = U.build_optimizer(self.hparams.optimizer, self.model)
        if self.hparams.lr_scheduler is not None:
            self.hparams.lr_scheduler.scheduler.total_steps = (
                self.trainer.estimated_stepping_batches
            )
            scheduler = U.build_scheduler(
                self.hparams.lr_scheduler.scheduler, optimizer=optimizer
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.lr_scheduler.get("monitor", "val/loss"),
                    "interval": self.hparams.lr_scheduler.get("interval", "step"),
                    "frequency": self.hparams.lr_scheduler.get("frequency", 1),
                },
            }
        return {"optimizer": optimizer}
