import logging
import os
import pathlib

import cv2
import torch
import rasterio as rio

from innofw.constants import Frameworks, Stages, SegOutKeys

from innofw.core.datamodules.lightning_datamodules.base import (
    BaseLightningDataModule,
)
from innofw.core.datasets.hdf5 import HDF5Dataset


class HDF5LightningDataModule(BaseLightningDataModule):
    """Class defines hdf5 dataset preparation and dataloader creation for semantic segmentation

        Attributes
        ----------
        task: List[str]
            the task the datamodule is intended to be used for
        framework: List[Union[str, Frameworks]]
            the model framework the datamodule is designed to work with

        Methods
        -------
        setup_train_test_val
            finds hdf5 files
            splits train data into train and validation sets
            creates dataset objects
        save_preds(preds, stage: Stages, dst_path: pathlib.Path)
            saves predicted segmentation masks as file in the destination folder
    """
    task = ["image-segmentation"]
    framework = [Frameworks.torch]

    def __init__(
            self,
            train,
            test,
            infer=None,
            augmentations=None,
            channels_num: int = 3,
            val_size: float = 0.2,
            batch_size: int = 32,
            num_workers: int = 1,
            random_seed: int = 42,
            threshold: float = 0.3,
            stage=None,
            *args,
            **kwargs,
    ):
        super().__init__(
            train=train,
            test=test,
            batch_size=batch_size,
            num_workers=num_workers,
            infer=infer,
            stage=stage,
            *args,
            **kwargs,
        )

        self.aug = augmentations
        self.channels_num = channels_num
        self.val_size = val_size
        self.random_seed = random_seed
        self.threshold = threshold

    def find_hdf5(self, path):
        paths = []
        if not os.path.isfile(path):
            for p in os.listdir(path):
                paths.append(os.path.join(path, p))
        return paths or [path]

    def setup_train_test_val(self, **kwargs):
        # files = list(self.data_path.rglob(''))
        train_files = self.find_hdf5(self.train_dataset)
        test_files = self.find_hdf5(self.test_dataset)

        # prepare datasets
        train_val = HDF5Dataset(train_files, self.channels_num, self.aug)
        val_size = int(len(train_val) * float(self.val_size))
        train, val = torch.utils.data.random_split(
            train_val, [len(train_val) - val_size, val_size]
        )

        self.train_dataset = train
        self.test_dataset = HDF5Dataset(test_files, self.channels_num, self.aug)
        self.val_dataset = val

    def setup_infer(self):
        if isinstance(self.predict_dataset, HDF5Dataset):
            return
        infer_files = self.find_hdf5(self.predict_dataset)
        self.predict_dataset = HDF5Dataset(infer_files, self.channels_num, self.aug)

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        out_file_path = dst_path / "results"
        os.mkdir(out_file_path)
        i = 0
        for preds_batch in preds:
            for pred, image, mask in zip(preds_batch[SegOutKeys.predictions], preds_batch[SegOutKeys.image], preds_batch[SegOutKeys.label]):
                pred = pred.numpy()
                image = image.numpy()
                mask = mask.numpy() * 255

                pred[pred < self.threshold] = 0  # todo: refactor
                pred[pred > 0] = 255  # make it suitable for multiclass/multilabel case
                format = '.tif'  # todo: refactor!!!!!
                if format == '.tif':
                    filename = out_file_path / f"out_{i}.tif"

                    with rio.open(filename, 'w', height=pred.shape[0], width=pred.shape[1], count=1, dtype='float32') as f:
                        f.write(pred, 1)

                    with rio.open(out_file_path / f"img_{i}.tif", 'w', height=image.shape[1], width=image.shape[2], count=image.shape[0], dtype='float32') as f:  # todo: refactor: use metadata
                        f.write(image)

                    with rio.open(out_file_path / f"mask_{i}.tif", 'w', height=mask.shape[1], width=mask.shape[2], count=1, dtype='float32') as f:  # todo: refactor
                        f.write(mask.squeeze(), 1)  # todo: refactor: varying channels
                else:
                    pass
                    # cv2.imwrite(filename, pred)  # [0]
                i += 1
        logging.info(f"Saved result to: {out_file_path}")
