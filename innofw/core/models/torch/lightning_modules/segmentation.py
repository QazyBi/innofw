__all__ = ["SemanticSegmentationLightningModule"]

# standard libraries
import logging
from typing import Any, Optional

# third-party libraries
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryRecall,
    BinaryPrecision,
    BinaryJaccardIndex,
)
from torch.cuda.amp import GradScaler, autocast
import torch
from torchmetrics import MetricCollection
# import lovely_tensors as lt
from torch import Tensor


# local modules
from innofw.constants import SegDataKeys, SegOutKeys


# lt.monkey_patch()


class SemanticSegmentationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        loss,
        optim_config,
        metrics=None,
        scheduler_cfg=None,
        threshold=0.5,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(model, DictConfig):
            self.model = hydra.utils.instantiate(model)
        else:
            self.model = model

        self.losses = loss  # hydra.utils.instantiate(loss)
        self.optim_config = optim_config
        self.scheduler_cfg = scheduler_cfg
        self.threshold = threshold
        # self.automatic_optimization = False
        metrics = MetricCollection(
            [
                BinaryF1Score(threshold=threshold),
                BinaryPrecision(threshold=threshold),
                BinaryRecall(threshold=threshold),
                BinaryJaccardIndex(threshold=threshold),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        self.scaler = GradScaler(enabled=True)
        self.save_hyperparameters(ignore=["metrics", "optim_config", "scheduler_cfg"])

    def forward(self, raster):
        return torch.max(self.model(raster), dim=1)[0]

    def configure_optimizers(self):
        output = {}

        # instantiate the optimizer
        optimizer = hydra.utils.instantiate(
            self.optim_config, params=self.model.parameters()
        )
        output["optimizer"] = optimizer

        if self.scheduler_cfg is not None:
            # instantiate the scheduler
            scheduler = hydra.utils.instantiate(self.scheduler_cfg, optimizer=optimizer)
            output["lr_scheduler"] = scheduler
        return output

    # def backward(
    #     self,
    #     loss: Tensor,
    #     optimizer,
    #     optimizer_idx: Optional[int],
    #     *args,
    #     **kwargs,
    # ) -> None:
    #     # return super().backward(loss, optimizer, optimizer_idx, *args, **kwargs):
    #     res = self.scaler.scale(loss).backward()
    #     self.scaler.step(optimizer)
    #     self.scaler.update()
    #     self.lr_schedulers().step()
    #     torch.cuda.synchronize()
    #     return res

    # def compute_loss(self, predictions, labels):
    #     loss = self.loss(predictions, labels)

    #     # with autocast(enabled=train_cfg["AMP"]):
    #     #     logits = model(img)
    #     #     loss = loss_fn(logits, lbl)
    #     return loss
    #     # return self.scaler.scale(loss)  # todo: refactor !!!!

    def compute_metrics(self, stage, predictions, labels):
        if stage == "train":
            return self.train_metrics(predictions.view(-1), labels.view(-1))
        elif stage == "val":
            out1 = self.val_metrics(predictions.view(-1), labels.view(-1))
            return out1
        elif stage == "test":
            return self.test_metrics(predictions.view(-1), labels.view(-1))

    def log_losses(
            self, name: str, logits: torch.Tensor, masks: torch.Tensor
    ) -> torch.FloatTensor:
        """Function to compute and log losses"""
        total_loss = 0
        for loss_name, weight, loss in self.losses:
            # for loss_name in loss_dict:
            ls_mask = loss(logits, masks)
            total_loss += weight * ls_mask

            self.log(
                f"loss/{name}/{weight} * {loss_name}",
                ls_mask,
                on_step=False,
                on_epoch=True,
            )

        self.log(f"loss/{name}", total_loss, on_step=False, on_epoch=True)
        return total_loss

    def log_metrics(self, stage, metrics_res):
        for key, value in metrics_res.items():
            self.log(key, value, sync_dist=True)

    def stage_step(self, stage, batch, do_logging=False, *args, **kwargs):
        output = dict()
        self.optimizers().zero_grad()  # set_to_none=True
        # todo: check that model is in mode no autograd
        raster, label = batch[SegDataKeys.image], batch[SegDataKeys.label]

        # with autocast(enabled=True):
        predictions = self.forward(raster)
        # if (
        #     predictions.max() > 1 or predictions.min() < 0
        # ):  # todo: should be configurable via cfg file
        #     predictions = torch.sigmoid(predictions)

        output[SegOutKeys.predictions] = predictions

        if stage in ["train", "val"]:
            # with autocast(enabled=True):
            loss = self.log_losses(stage, predictions.squeeze(), label.squeeze())
                # loss = self.compute_loss(predictions, label)
            # self.log_losses(stage, loss)
            output["loss"] = loss

            # self.scaler.scale(loss).backward()
            # self.scaler.step(self.optimizers)
            # self.scaler.update()
            # self.lr_schedulers().step()
            # torch.cuda.synchronize()

        # if stage != "predict":
        #     metrics = self.compute_metrics(stage, predictions, label)  # todo: uncomment
        #     self.log_metrics(stage, metrics)
        # torch.cuda.empty_cache()

        return output

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        return self.stage_step("train", batch, do_logging=True)

    def validation_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.stage_step("val", batch)

    def test_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.stage_step("test", batch)

    # def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
    #     tile, coords = batch[SegDataKeys.image], batch[SegDataKeys.coords]
    #
    #     prediction = self.forward(tile)
    #     if dataloader_idx is None:
    #         self.trainer.predict_dataloaders[0].dataset.add_prediction(prediction, coords, batch_idx)
    #     else:
    #         self.trainer.predict_dataloaders[dataloader_idx].dataset.add_prediction(prediction, coords, batch_idx)