from typing import Optional

import numpy as np
import torch
from innofw.constants import SegDataKeys
import logging


def prep_data(
    image, mask: Optional = None, transform: Optional = None
):
    if transform is not None:
        if mask is not None:
            sample = transform(image=image.astype("uint8"), mask=mask.astype("uint8"))
            image, mask = sample["image"], sample["mask"]
        else:
            sample = transform(image=image.astype("uint8"))
            image = sample["image"]

    image = np.moveaxis(image, 2, 0)
    logging.debug("doing preprocessing: division by 255")
    # ============== preprocessing ==============
    image = image / 255.0
    # ===========================================
    image = torch.from_numpy(image)
    image = image.float()
    if mask is not None:
        mask = (mask > 0).astype(np.uint8)
        mask = torch.from_numpy(mask.copy())
        mask = torch.unsqueeze(mask, 0).float()

        return {SegDataKeys.image: image, SegDataKeys.label: mask}

    return {SegDataKeys.image: image}
